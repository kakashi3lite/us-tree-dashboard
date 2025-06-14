import React, { createContext, useContext, useState, useCallback } from 'react';
import axios from 'axios';

const GeospatialContext = createContext();

export const useGeospatialContext = () => {
  const context = useContext(GeospatialContext);
  if (!context) {
    throw new Error('useGeospatialContext must be used within a GeospatialProvider');
  }
  return context;
};

export const GeospatialProvider = ({ children }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchTreesInRadius = useCallback(async (lat, lon, radius) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get('/api/trees/radius', {
        params: { lat, lon, radius }
      });
      return response.data;
    } catch (err) {
      setError('Failed to fetch trees in radius');
      console.error('Error fetching trees in radius:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchSpeciesDistribution = useCallback(async (bbox) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get('/api/trees/species-distribution', {
        params: { bbox: bbox.join(',') }
      });
      return response.data;
    } catch (err) {
      setError('Failed to fetch species distribution');
      console.error('Error fetching species distribution:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchTreeDensityHeatmap = useCallback(async (bbox) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get('/api/trees/density-heatmap', {
        params: { bbox: bbox.join(',') }
      });
      return response.data;
    } catch (err) {
      setError('Failed to fetch tree density heatmap');
      console.error('Error fetching tree density heatmap:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchPredictedSpeciesDistribution = useCallback(async (bbox, climateScenario) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get('/api/predictions/species-distribution', {
        params: {
          bbox: bbox.join(','),
          climate_scenario: climateScenario
        }
      });
      return response.data;
    } catch (err) {
      setError('Failed to fetch predicted species distribution');
      console.error('Error fetching predicted species distribution:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchClimateImpactAnalysis = useCallback(async (bbox, timeRange) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get('/api/analysis/climate-impact', {
        params: {
          bbox: bbox.join(','),
          start_date: timeRange.start,
          end_date: timeRange.end
        }
      });
      return response.data;
    } catch (err) {
      setError('Failed to fetch climate impact analysis');
      console.error('Error fetching climate impact analysis:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const value = {
    loading,
    error,
    fetchTreesInRadius,
    fetchSpeciesDistribution,
    fetchTreeDensityHeatmap,
    fetchPredictedSpeciesDistribution,
    fetchClimateImpactAnalysis
  };

  return (
    <GeospatialContext.Provider value={value}>
      {children}
    </GeospatialContext.Provider>
  );
};
