/**
 * Calculate marker size based on tree count and device type
 * @param {number} count - Number of trees in cluster
 * @param {boolean} isMobile - Whether the device is mobile
 * @returns {number} Marker size in pixels
 */
export const calculateMarkerSize = (count, isMobile) => {
  const baseSize = isMobile ? 4 : 6;
  const maxSize = isMobile ? 15 : 25;
  const size = Math.log(count + 1) * baseSize;
  return Math.min(size, maxSize);
};

/**
 * Get marker color based on tree health conditions
 * @param {Object} healthConditions - Object containing health condition counts
 * @returns {string} Hex color code
 */
export const getColorByHealth = (healthConditions) => {
  const total = Object.values(healthConditions).reduce((a, b) => a + b, 0);
  const healthyCount = healthConditions['Healthy'] || 0;
  const healthyRatio = healthyCount / total;

  if (healthyRatio >= 0.8) return '#4caf50';  // Mostly healthy - Green
  if (healthyRatio >= 0.5) return '#ffeb3b';  // Mixed health - Yellow
  return '#f44336';  // Poor health - Red
};

/**
 * Format cluster popup content
 * @param {Object} cluster - Tree cluster data
 * @returns {string} Formatted HTML content
 */
export const formatClusterPopup = (cluster) => {
  const { tree_count, species, health_conditions } = cluster;
  const healthStats = Object.entries(health_conditions)
    .map(([condition, count]) => `${condition}: ${count}`)
    .join('<br>');

  return `
    <div class="cluster-popup">
      <h3>${tree_count} Trees</h3>
      <p><strong>Species:</strong> ${species.length}</p>
      <p><strong>Health Distribution:</strong><br>${healthStats}</p>
    </div>
  `;
};

/**
 * Calculate viewport bounds from center point
 * @param {Array} center - [latitude, longitude]
 * @param {number} radiusKm - Radius in kilometers
 * @returns {Object} Bounds object with min/max lat/lon
 */
export const calculateViewportBounds = (center, radiusKm) => {
  const [lat, lon] = center;
  const latChange = (radiusKm / 111.32);  // 1 degree latitude = 111.32 km
  const lonChange = (radiusKm / (111.32 * Math.cos(lat * Math.PI / 180)));

  return {
    minLat: lat - latChange,
    maxLat: lat + latChange,
    minLon: lon - lonChange,
    maxLon: lon + lonChange
  };
};

/**
 * Check if a point is within the current viewport
 * @param {Array} point - [latitude, longitude]
 * @param {Object} bounds - Viewport bounds
 * @returns {boolean} Whether point is within bounds
 */
export const isPointInViewport = (point, bounds) => {
  const [lat, lon] = point;
  return lat >= bounds.minLat &&
         lat <= bounds.maxLat &&
         lon >= bounds.minLon &&
         lon <= bounds.maxLon;
};
