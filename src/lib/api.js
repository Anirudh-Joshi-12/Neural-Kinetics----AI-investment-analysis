// src/lib/api.js
const API_BASE = 'http://localhost:8001';

async function handleResponse(response) {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  // 204 No Content has no body to parse
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

async function handleBlobResponse(response) {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.blob();
}

export const api = {
  async uploadReport(file) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE}/reports/upload`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(response);
  },

  async listReports(filters = {}) {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== '' && value !== null && value !== undefined) {
        params.append(key, value);
      }
    });
    const response = await fetch(`${API_BASE}/reports?${params}`);
    return handleResponse(response);
  },

  async getReport(id) {
    const response = await fetch(`${API_BASE}/reports/${id}`);
    return handleResponse(response);
  },

  async deleteReport(id) {
    const response = await fetch(`${API_BASE}/reports/${id}`, {
      method: 'DELETE',
    });
    return handleResponse(response);
  },

  async getReportStatus(id) {
    const response = await fetch(`${API_BASE}/reports/${id}/status`);
    return handleResponse(response);
  },

  async downloadReportPdf(id) {
    const response = await fetch(`${API_BASE}/reports/${id}/pdf`);
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `report-${id}.pdf`;
    a.click();
    URL.revokeObjectURL(url);
  },

  async reprocessReport(id) {
    const response = await fetch(`${API_BASE}/reports/${id}/reprocess`, {
      method: 'POST',
    });
    return handleResponse(response);
  },
};