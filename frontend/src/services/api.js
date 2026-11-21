import axios from 'axios';

const API_BASE_URL = 'http://localhost:5001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const schedulerService = {
  simulate: (data) => api.post('/simulate', data),
  getCompare: () => api.get('/compare'),
  compareProcesses: (data, options = {}) => api.post('/compare', data, options),
  getLSTMPredictions: () => api.get('/lstm-predictions'),
  checkStatus: () => api.get('/'),
  // Real-time workloads
  periodicWorkload: (tasks, numCores = 1) => api.post('/workload/periodic', { tasks, num_cores: numCores }),
  addDeadlines: (processes, slack = 3) => api.post('/workload/deadlines', { processes, slack }),
  importTrace: (file, options = {}) => {
    const form = new FormData();
    form.append('file', file);
    Object.entries(options).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') form.append(k, String(v));
    });
    return api.post('/import-trace', form, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
};

export default api;
