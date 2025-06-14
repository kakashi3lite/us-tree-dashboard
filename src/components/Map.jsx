import React, { useEffect, useRef, useState, useMemo } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useQuery } from 'react-query';
import { debounce } from 'lodash';
import { useMediaQuery } from '@mui/material';
import { fetchTreeClusters } from '../services/api';
import { calculateMarkerSize, getColorByHealth } from '../utils/mapUtils';

const MapControls = ({ onBoundsChange, onZoomChange }) => {
  const map = useMap();

  useEffect(() => {
    const handleMoveEnd = debounce(() => {
      const bounds = map.getBounds();
      onBoundsChange({
        minLat: bounds.getSouth(),
        maxLat: bounds.getNorth(),
        minLon: bounds.getWest(),
        maxLon: bounds.getEast()
      });
    }, 300);

    const handleZoomEnd = debounce(() => {
      onZoomChange(map.getZoom());
    }, 300);

    map.on('moveend', handleMoveEnd);
    map.on('zoomend', handleZoomEnd);

    return () => {
      map.off('moveend', handleMoveEnd);
      map.off('zoomend', handleZoomEnd);
    };
  }, [map, onBoundsChange, onZoomChange]);

  return null;
};

const TreeMap = ({ height = '70vh', onClusterClick }) => {
  const [bounds, setBounds] = useState(null);
  const [zoom, setZoom] = useState(12);
  const isMobile = useMediaQuery('(max-width:600px)');
  const mapRef = useRef(null);

  const mapStyle = useMemo(() => ({
    height,
    width: '100%',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  }), [height]);

  const { data: clusters = [], isLoading, error } = useQuery(
    ['treeClusters', bounds, zoom],
    () => fetchTreeClusters(bounds, zoom),
    {
      enabled: !!bounds,
      keepPreviousData: true,
      staleTime: 30000,
      cacheTime: 300000
    }
  );

  const handleBoundsChange = (newBounds) => {
    setBounds(newBounds);
  };

  const handleZoomChange = (newZoom) => {
    setZoom(newZoom);
  };

  const renderCluster = (cluster) => {
    const { centroid, tree_count, species, health_conditions } = cluster;
    const size = calculateMarkerSize(tree_count, isMobile);
    const color = getColorByHealth(health_conditions);

    return (
      <CircleMarker
        key={`${centroid.coordinates[0]}-${centroid.coordinates[1]}`}
        center={[centroid.coordinates[1], centroid.coordinates[0]]}
        radius={size}
        fillColor={color}
        color={color}
        weight={2}
        opacity={0.8}
        fillOpacity={0.6}
        eventHandlers={{
          click: () => onClusterClick?.(cluster)
        }}
      >
        <Popup>
          <div className="cluster-popup">
            <h3>{tree_count} Trees</h3>
            <p>Species: {species.length}</p>
            <p>Health: {Object.entries(health_conditions)
              .map(([condition, count]) => `${condition}: ${count}`)
              .join(', ')}</p>
          </div>
        </Popup>
      </CircleMarker>
    );
  };

  if (error) {
    console.error('Error loading tree clusters:', error);
    return <div>Error loading map data</div>;
  }

  return (
    <MapContainer
      ref={mapRef}
      center={[40.7128, -74.0060]}
      zoom={zoom}
      style={mapStyle}
      zoomControl={!isMobile}
      scrollWheelZoom={true}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      <MapControls
        onBoundsChange={handleBoundsChange}
        onZoomChange={handleZoomChange}
      />
      {clusters.map(renderCluster)}
    </MapContainer>
  );
};

export default TreeMap;