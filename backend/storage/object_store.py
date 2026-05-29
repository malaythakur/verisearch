"""S3-compatible object store for cleaned document text.

Key layout convention:
    {tenant_id}/{document_id}/{version}/cleaned.txt
    _global/{document_id}/{version}/cleaned.txt  (for null tenant_id)

The cleaned_text_uri stored in document_versions uses the format:
    s3://{bucket}/{key}

The module is usable without a live S3 connection for key-building/parsing functions.
boto3 is imported lazily only when S3 operations are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.config.settings import Settings

if TYPE_CHECKING:
    import boto3

# Sentinel prefix for documents without a tenant (global crawl-derived docs)
_GLOBAL_PREFIX = "_global"

# Fixed filename for cleaned text objects
_CLEANED_FILENAME = "cleaned.txt"


class DocumentObjectStore:
    """Manages S3 storage of cleaned document text with content-addressed keys.

    Key format: {tenant_id}/{document_id}/{version}/cleaned.txt
    For global documents (tenant_id is None): _global/{document_id}/{version}/cleaned.txt

    Objects are immutable per version, supporting idempotent re-indexing (R2.3/R2.4).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the object store.

        Args:
            settings: Application settings. If None, loads from environment.
        """
        self._settings = settings or Settings()
        self._client: object | None = None

    @staticmethod
    def build_key(tenant_id: str | None, document_id: str, version: int) -> str:
        """Construct the S3 object key for a cleaned text document.

        Args:
            tenant_id: The tenant identifier, or None for global documents.
            document_id: The document identifier (UUID string).
            version: The document version (positive integer).

        Returns:
            The S3 key string.

        Raises:
            ValueError: If document_id is empty, or version is less than 1.
        """
        if not document_id:
            raise ValueError("document_id must not be empty")
        if version < 1:
            raise ValueError("version must be >= 1")

        prefix = tenant_id if tenant_id is not None else _GLOBAL_PREFIX
        return f"{prefix}/{document_id}/{version}/{_CLEANED_FILENAME}"

    @staticmethod
    def parse_key(key: str) -> tuple[str | None, str, int]:
        """Parse an S3 key back into its component parts.

        Args:
            key: The S3 key string in the format
                 {tenant_id}/{document_id}/{version}/cleaned.txt

        Returns:
            A tuple of (tenant_id, document_id, version) where tenant_id is None
            for global documents.

        Raises:
            ValueError: If the key does not match the expected format.
        """
        parts = key.split("/")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid key format: expected 4 path segments, got {len(parts)}. "
                f"Key must be '{{tenant_id}}/{{document_id}}/{{version}}/{_CLEANED_FILENAME}'"
            )

        prefix, document_id, version_str, filename = parts

        if filename != _CLEANED_FILENAME:
            raise ValueError(
                f"Invalid key format: expected filename '{_CLEANED_FILENAME}', got '{filename}'"
            )

        if not document_id:
            raise ValueError("Invalid key format: document_id segment is empty")

        try:
            version = int(version_str)
        except ValueError:
            raise ValueError(
                f"Invalid key format: version segment '{version_str}' is not a valid integer"
            )

        if version < 1:
            raise ValueError(f"Invalid key format: version must be >= 1, got {version}")

        tenant_id = None if prefix == _GLOBAL_PREFIX else prefix
        return (tenant_id, document_id, version)

    def _build_uri(self, key: str) -> str:
        """Build the full S3 URI for a key.

        Args:
            key: The S3 object key.

        Returns:
            The full s3:// URI.
        """
        return f"s3://{self._settings.s3_bucket}/{key}"

    def _parse_uri(self, uri: str) -> str:
        """Extract the S3 key from a full URI.

        Args:
            uri: The full s3:// URI.

        Returns:
            The S3 object key.

        Raises:
            ValueError: If the URI format is invalid.
        """
        expected_prefix = f"s3://{self._settings.s3_bucket}/"
        if not uri.startswith(expected_prefix):
            raise ValueError(
                f"Invalid URI: expected prefix '{expected_prefix}', got '{uri}'"
            )
        return uri[len(expected_prefix):]

    def _get_client(self):
        """Get or create the boto3 S3 client (lazy initialization).

        boto3 is imported here to allow key-building/parsing to work
        without boto3 installed.
        """
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=self._settings.aws_endpoint_url,
                aws_access_key_id=self._settings.aws_access_key_id,
                aws_secret_access_key=self._settings.aws_secret_access_key,
                region_name=self._settings.aws_default_region,
            )
        return self._client

    async def put_cleaned_text(
        self, tenant_id: str | None, document_id: str, version: int, content: str
    ) -> str:
        """Store cleaned text content in S3.

        Args:
            tenant_id: The tenant identifier, or None for global documents.
            document_id: The document identifier.
            version: The document version.
            content: The cleaned text content to store.

        Returns:
            The full S3 URI (s3://bucket/key) for the stored object.
        """
        key = self.build_key(tenant_id, document_id, version)
        client = self._get_client()
        client.put_object(
            Bucket=self._settings.s3_bucket,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
        return self._build_uri(key)

    async def get_cleaned_text(self, uri: str) -> str:
        """Retrieve cleaned text content from S3 by URI.

        Args:
            uri: The full S3 URI (s3://bucket/key).

        Returns:
            The cleaned text content as a string.

        Raises:
            FileNotFoundError: If the object does not exist.
            ValueError: If the URI format is invalid.
        """
        from botocore.exceptions import ClientError

        key = self._parse_uri(uri)
        client = self._get_client()
        try:
            response = client.get_object(
                Bucket=self._settings.s3_bucket,
                Key=key,
            )
            body = response["Body"].read()
            return body.decode("utf-8")
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Object not found: {uri}") from e
            raise

    async def exists(
        self, tenant_id: str | None, document_id: str, version: int
    ) -> bool:
        """Check if cleaned text exists for the given document version.

        Args:
            tenant_id: The tenant identifier, or None for global documents.
            document_id: The document identifier.
            version: The document version.

        Returns:
            True if the object exists, False otherwise.
        """
        from botocore.exceptions import ClientError

        key = self.build_key(tenant_id, document_id, version)
        client = self._get_client()
        try:
            client.head_object(
                Bucket=self._settings.s3_bucket,
                Key=key,
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    async def delete_tenant_data(self, tenant_id: str) -> int:
        """Delete all objects for a given tenant.

        Used for tenant data deletion requests. Iterates over all objects
        with the tenant's prefix and deletes them in batches.

        Args:
            tenant_id: The tenant identifier. Must not be None.

        Returns:
            The number of objects deleted.

        Raises:
            ValueError: If tenant_id is None or empty.
        """
        if not tenant_id:
            raise ValueError("tenant_id must not be empty for deletion")

        client = self._get_client()
        prefix = f"{tenant_id}/"
        deleted_count = 0

        # List and delete objects in batches (S3 delete_objects supports up to 1000)
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self._settings.s3_bucket, Prefix=prefix
        ):
            contents = page.get("Contents", [])
            if not contents:
                continue

            # Delete in batches of up to 1000
            objects_to_delete = [{"Key": obj["Key"]} for obj in contents]
            client.delete_objects(
                Bucket=self._settings.s3_bucket,
                Delete={"Objects": objects_to_delete, "Quiet": True},
            )
            deleted_count += len(objects_to_delete)

        return deleted_count
