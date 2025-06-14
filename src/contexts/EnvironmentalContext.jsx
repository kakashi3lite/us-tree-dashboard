import React, { createContext, useContext, useState, useCallback } from 'react';
import axios from 'axios';

const EnvironmentalContext = createContext();

export const useEnvironmentalContext = () => {
  const context = useContext(EnvironmentalContext);
  if (!context) {
    throw new Error('useEnvironmentalContext must be used within an EnvironmentalProvider');
  }
  return context;
};

export const EnvironmentalProvider = ({ children }) => {
  const [environmentalMetrics, setEnvironmentalMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const updateEnvironmentalMetrics = useCallback((data) => {
    setEnvironmentalMetrics(data);
  }, []);

  const fetchEnvironmentalImpact = useCallback(async (bbox) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get('/api/environmental/impact', {
        params: { bbox: bbox.join(',') }
      });
      setEnvironmentalMetrics(response.data);
      return response.data;
    } catch (err) {
      setError('Failed to fetch environmental impact');
      console.error('Error fetching environmental impact:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchHistoricalTrends = useCallback(async (bbox, timeRange) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get('/api/environmental/historical-trends', {
        params: {
          bbox: bbox.join(','),
          start_date: timeRange.start,
          end_date: timeRange.end
        }
      });
      return response.data;
    } catch (err) {
      setError('Failed to fetch historical trends');
      console.error('Error fetching historical trends:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchClimateScenarios = useCallback(async (bbox) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get('/api/environmental/climate-scenarios', {
        params: { bbox: bbox.join(',') }
      });
      return response.data;
    } catch (err) {
      setError('Failed to fetch climate scenarios');
      console.error('Error fetching climate scenarios:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const value = {
    environmentalMetrics,
    loading,
    error,
    updateEnvironmentalMetrics,
    fetchEnvironmentalImpact,
    fetchHistoricalTrends,
    fetchClimateScenarios
  };

  return (
    <EnvironmentalContext.Provider value={value}>
      {children}
    </EnvironmentalContext.Provider>
  );
};
