{{/*
Expand the name of the chart.
*/}}
{{- define "agentic-research.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "agentic-research.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "agentic-research.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "agentic-research.labels" -}}
helm.sh/chart: {{ include "agentic-research.chart" . }}
{{ include "agentic-research.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: agentic-research-engine
{{- end }}

{{/*
Selector labels
*/}}
{{- define "agentic-research.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agentic-research.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "agentic-research.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "agentic-research.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
OTLP endpoint
*/}}
{{- define "agentic-research.otlpEndpoint" -}}
{{- .Values.observability.tracing.otlpEndpoint | default "http://otel-collector:4317" }}
{{- end }}
