import axios from 'axios';
import { handleApiError } from '../utils/errorHandlers';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8050/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor for API calls
api.interceptors.request.use(
  async config => {
    // Add any request preprocessing here (e.g., auth tokens)
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// Response interceptor for API calls
api.interceptors.response.use(
  response => response.data,
  error => handleApiError(error)
);

export const fetchTreeClusters = async (bounds, zoom) => {
  try {
    const response = await api.get('/tree-clusters', {
      params: {
        min_lat: bounds.minLat,
        max_lat: bounds.maxLat,
        min_lon: bounds.minLon,
        max_lon: bounds.maxLon,
        zoom
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching tree clusters:', error);
    throw error;
  }
};

export const fetchTreeDensity = async (regionId = null) => {
  try {
    const response = await api.get('/tree-density', {
      params: { region_id: regionId }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching tree density:', error);
    throw error;
  }
};

export const fetchEnvironmentalImpact = async (regionId = null, startDate = null, endDate = null) => {
  try {
    const response = await api.get('/environmental-impact', {
      params: {
        region_id: regionId,
        start_date: startDate?.toISOString().split('T')[0],
        end_date: endDate?.toISOString().split('T')[0]
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching environmental impact:', error);
    throw error;
  }
};

export const fetchHistoricalTrends = async (
  metric,
  interval = 'month',
  startDate = null,
  endDate = null
) => {
  try {
    const response = await api.get('/historical-trends', {
      params: {
        metric,
        interval,
        start_date: startDate?.toISOString().split('T')[0],
        end_date: endDate?.toISOString().split('T')[0]
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching historical trends:', error);
    throw error;
  }
};

export const retryFailedRequest = async (failedRequest, maxRetries = 3) => {
  let retries = 0;
  while (retries < maxRetries) {
    try {
      return await failedRequest();
    } catch (error) {
      retries++;
      if (retries === maxRetries) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * retries));
    }
  }
};
