/**
 * Utility functions for map operations and data processing
 */

import { scaleLinear } from 'd3-scale';
import { extent } from 'd3-array';

/**
 * Calculate the bounding box from a set of coordinates
 * @param {Array} coordinates Array of [lon, lat] coordinates
 * @returns {Array} [minLon, minLat, maxLon, maxLat]
 */
export const calculateBoundingBox = (coordinates) => {
  const lons = coordinates.map(coord => coord[0]);
  const lats = coordinates.map(coord => coord[1]);
  
  return [
    Math.min(...lons),
    Math.min(...lats),
    Math.max(...lons),
    Math.max(...lats)
  ];
};

/**
 * Generate a color scale for tree density visualization
 * @param {Array} data Array of density values
 * @returns {Function} Color scale function
 */
export const createDensityColorScale = (data) => {
  const [min, max] = extent(data);
  
  return scaleLinear()
    .domain([min, max])
    .range(['#c6dbef', '#08519c']);
};

/**
 * Calculate the optimal zoom level for a bounding box
 * @param {Array} bbox [minLon, minLat, maxLon, maxLat]
 * @param {Object} mapSize {width, height} in pixels
 * @returns {number} Optimal zoom level
 */
export const calculateOptimalZoom = (bbox, mapSize) => {
  const WORLD_WIDTH = 360;
  const PADDING = 0.1;
  
  const lonSpan = bbox[2] - bbox[0];
  const latSpan = bbox[3] - bbox[1];
  
  const latZoom = Math.floor(Math.log2(mapSize.height / 256 / latSpan));
  const lonZoom = Math.floor(Math.log2(mapSize.width / 256 / lonSpan));
  
  return Math.min(latZoom, lonZoom) - PADDING;
};

/**
 * Format tree data for cluster visualization
 * @param {Array} trees Array of tree objects
 * @returns {Object} GeoJSON FeatureCollection
 */
export const formatTreeClusters = (trees) => {
  return {
    type: 'FeatureCollection',
    features: trees.map(tree => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [tree.longitude, tree.latitude]
      },
      properties: {
        id: tree.id,
        species: tree.species,
        height: tree.height,
        diameter: tree.diameter,
        health: tree.health
      }
    }))
  };
};

/**
 * Calculate the grid size for density visualization based on zoom level
 * @param {number} zoom Current map zoom level
 * @returns {number} Grid size in kilometers
 */
export const calculateGridSize = (zoom) => {
  const BASE_GRID_SIZE = 10; // 10km at zoom level 10
  return BASE_GRID_SIZE * Math.pow(2, 10 - zoom);
};

/**
 * Generate a legend configuration for the current visualization
 * @param {Array} data Data array to generate legend for
 * @param {string} type Type of legend ('density', 'species', etc.)
 * @returns {Object} Legend configuration
 */
export const generateLegendConfig = (data, type) => {
  switch (type) {
    case 'density':
      const [min, max] = extent(data);
      const steps = 5;
      const stepSize = (max - min) / steps;
      
      return {
        title: 'Trees per km²',
        steps: Array.from({ length: steps + 1 }, (_, i) => {
          const value = min + (stepSize * i);
          return {
            value: Math.round(value),
            color: createDensityColorScale(data)(value)
          };
        })
      };
      
    case 'species':
      return {
        title: 'Tree Species',
        categories: data.map(species => ({
          label: species.name,
          color: species.color
        }))
      };
      
    default:
      return null;
  }
};

/**
 * Calculate the viewport padding based on screen size
 * @param {Object} screen {width, height} Screen dimensions
 * @returns {Object} Padding values for each side
 */
export const calculateViewportPadding = (screen) => {
  const BASE_PADDING = 50;
  const MOBILE_BREAKPOINT = 768;
  
  return {
    top: screen.width < MOBILE_BREAKPOINT ? BASE_PADDING / 2 : BASE_PADDING,
    right: screen.width < MOBILE_BREAKPOINT ? BASE_PADDING / 2 : BASE_PADDING,
    bottom: screen.width < MOBILE_BREAKPOINT ? BASE_PADDING / 2 : BASE_PADDING,
    left: screen.width < MOBILE_BREAKPOINT ? BASE_PADDING / 2 : BASE_PADDING
  };
};
