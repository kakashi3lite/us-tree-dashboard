import axios from 'axios';
import { handleError } from '../utils/errorHandling';

// Configure axios defaults
axios.defaults.baseURL = process.env.REACT_APP_API_URL || '';
axios.defaults.timeout = 30000; // 30 seconds

// Add request interceptor for authentication
axios.interceptors.request.use(
  config => {
    // Add any auth tokens or headers here
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
axios.interceptors.response.use(
  response => response,
  error => {
    handleError(error);
    return Promise.reject(error);
  }
);

// Tree-related API calls
export const getTreesInRadius = async (lat, lon, radius) => {
  try {
    const response = await axios.get('/api/trees/radius', {
      params: { lat, lon, radius }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching trees in radius:', error);
    throw error;
  }
};

export const getSpeciesDistribution = async (bbox) => {
  try {
    const response = await axios.get('/api/trees/species-distribution', {
      params: { bbox: bbox.join(',') }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching species distribution:', error);
    throw error;
  }
};

export const getTreeDensityHeatmap = async (bbox) => {
  try {
    const response = await axios.get('/api/trees/density-heatmap', {
      params: { bbox: bbox.join(',') }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching tree density heatmap:', error);
    throw error;
  }
};

// Prediction-related API calls
export const getPredictedSpeciesDistribution = async (bbox, climateScenario) => {
  try {
    const response = await axios.get('/api/predictions/species-distribution', {
      params: {
        bbox: bbox.join(','),
        climate_scenario: climateScenario
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching predicted species distribution:', error);
    throw error;
  }
};

// Environmental analysis API calls
export const getClimateImpactAnalysis = async (bbox, timeRange) => {
  try {
    const response = await axios.get('/api/analysis/climate-impact', {
      params: {
        bbox: bbox.join(','),
        start_date: timeRange.start,
        end_date: timeRange.end
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching climate impact analysis:', error);
    throw error;
  }
};

export const getEnvironmentalImpact = async (bbox) => {
  try {
    const response = await axios.get('/api/environmental/impact', {
      params: { bbox: bbox.join(',') }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching environmental impact:', error);
    throw error;
  }
};

export const getHistoricalTrends = async (bbox, timeRange) => {
  try {
    const response = await axios.get('/api/environmental/historical-trends', {
      params: {
        bbox: bbox.join(','),
        start_date: timeRange.start,
        end_date: timeRange.end
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching historical trends:', error);
    throw error;
  }
};

export const getClimateScenarios = async (bbox) => {
  try {
    const response = await axios.get('/api/environmental/climate-scenarios', {
      params: { bbox: bbox.join(',') }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching climate scenarios:', error);
    throw error;
  }
};
