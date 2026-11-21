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
};

export default api;
