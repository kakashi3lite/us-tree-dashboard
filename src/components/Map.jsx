import React, { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { debounce } from 'lodash';
import { useGeospatialContext } from '../contexts/GeospatialContext';
import { useEnvironmentalContext } from '../contexts/EnvironmentalContext';
import { MAPBOX_TOKEN } from '../config/settings';

const Map = () => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [viewport, setViewport] = useState({
    lng: -98.5795,
    lat: 39.8283,
    zoom: 4
  });

  const { 
    fetchTreesInRadius,
    fetchSpeciesDistribution,
    fetchTreeDensityHeatmap
  } = useGeospatialContext();

  const { 
    environmentalMetrics,
    updateEnvironmentalMetrics 
  } = useEnvironmentalContext();

  useEffect(() => {
    if (map.current) return;

    mapboxgl.accessToken = MAPBOX_TOKEN;
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/light-v11',
      center: [viewport.lng, viewport.lat],
      zoom: viewport.zoom,
      maxZoom: 18
    });

    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

    // Add custom layers
    map.current.on('load', () => {
      // Tree density heatmap layer
      map.current.addSource('tree-density', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: []
        }
      });

      map.current.addLayer({
        id: 'tree-density-heat',
        type: 'heatmap',
        source: 'tree-density',
        paint: {
          'heatmap-weight': [
            'interpolate',
            ['linear'],
            ['get', 'tree_count'],
            0, 0,
            100, 1
          ],
          'heatmap-intensity': [
            'interpolate',
            ['linear'],
            ['zoom'],
            0, 1,
            15, 3
          ],
          'heatmap-color': [
            'interpolate',
            ['linear'],
            ['heatmap-density'],
            0, 'rgba(33,102,172,0)',
            0.2, 'rgb(103,169,207)',
            0.4, 'rgb(209,229,240)',
            0.6, 'rgb(253,219,199)',
            0.8, 'rgb(239,138,98)',
            1, 'rgb(178,24,43)'
          ],
          'heatmap-radius': [
            'interpolate',
            ['linear'],
            ['zoom'],
            0, 2,
            15, 20
          ],
          'heatmap-opacity': 0.8
        }
      });

      // Individual trees layer
      map.current.addSource('trees', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: []
        },
        cluster: true,
        clusterMaxZoom: 14,
        clusterRadius: 50
      });

      map.current.addLayer({
        id: 'tree-clusters',
        type: 'circle',
        source: 'trees',
        filter: ['has', 'point_count'],
        paint: {
          'circle-color': [
            'step',
            ['get', 'point_count'],
            '#51bbd6',
            100,
            '#f1f075',
            750,
            '#f28cb1'
          ],
          'circle-radius': [
            'step',
            ['get', 'point_count'],
            20,
            100,
            30,
            750,
            40
          ]
        }
      });

      map.current.addLayer({
        id: 'tree-cluster-count',
        type: 'symbol',
        source: 'trees',
        filter: ['has', 'point_count'],
        layout: {
          'text-field': '{point_count_abbreviated}',
          'text-font': ['DIN Offc Pro Medium', 'Arial Unicode MS Bold'],
          'text-size': 12
        }
      });

      map.current.addLayer({
        id: 'unclustered-trees',
        type: 'circle',
        source: 'trees',
        filter: ['!', ['has', 'point_count']],
        paint: {
          'circle-color': '#11b4da',
          'circle-radius': 6,
          'circle-stroke-width': 1,
          'circle-stroke-color': '#fff'
        }
      });
    });

    // Update viewport on map move
    map.current.on('moveend', () => {
      const center = map.current.getCenter();
      const zoom = map.current.getZoom();
      setViewport({
        lng: center.lng,
        lat: center.lat,
        zoom: zoom
      });
      updateMapData();
    });

    return () => map.current.remove();
  }, []);

  // Debounced function to update map data
  const updateMapData = debounce(async () => {
    if (!map.current) return;

    const bounds = map.current.getBounds();
    const bbox = [
      bounds.getWest(),
      bounds.getSouth(),
      bounds.getEast(),
      bounds.getNorth()
    ];

    try {
      // Update tree density heatmap
      const heatmapData = await fetchTreeDensityHeatmap(bbox);
      if (map.current.getSource('tree-density')) {
        map.current.getSource('tree-density').setData(heatmapData);
      }

      // Update individual trees if zoom level is high enough
      if (map.current.getZoom() > 12) {
        const center = map.current.getCenter();
        const treeData = await fetchTreesInRadius(
          center.lat,
          center.lng,
          5 // 5km radius
        );
        if (map.current.getSource('trees')) {
          map.current.getSource('trees').setData(treeData);
        }
      }

      // Update species distribution
      const speciesData = await fetchSpeciesDistribution(bbox);
      updateEnvironmentalMetrics(speciesData);

    } catch (error) {
      console.error('Error updating map data:', error);
    }
  }, 500);

  // Handle tree cluster click
  const handleClusterClick = (e) => {
    const features = map.current.queryRenderedFeatures(e.point, {
      layers: ['tree-clusters']
    });
    const clusterId = features[0].properties.cluster_id;
    map.current.getSource('trees').getClusterExpansionZoom(
      clusterId,
      (err, zoom) => {
        if (err) return;

        map.current.easeTo({
          center: features[0].geometry.coordinates,
          zoom: zoom
        });
      }
    );
  };

  // Handle individual tree click
  const handleTreeClick = (e) => {
    const features = map.current.queryRenderedFeatures(e.point, {
      layers: ['unclustered-trees']
    });
    if (!features.length) return;

    const { properties } = features[0];
    new mapboxgl.Popup()
      .setLngLat(features[0].geometry.coordinates)
      .setHTML(`
        <h3>${properties.species}</h3>
        <p>Height: ${properties.height}m</p>
        <p>Diameter: ${properties.diameter}cm</p>
        <p>Health: ${properties.health}</p>
      `)
      .addTo(map.current);
  };

  useEffect(() => {
    if (!map.current) return;

    map.current.on('click', 'tree-clusters', handleClusterClick);
    map.current.on('click', 'unclustered-trees', handleTreeClick);

    map.current.on('mouseenter', 'tree-clusters', () => {
      map.current.getCanvas().style.cursor = 'pointer';
    });
    map.current.on('mouseleave', 'tree-clusters', () => {
      map.current.getCanvas().style.cursor = '';
    });

    map.current.on('mouseenter', 'unclustered-trees', () => {
      map.current.getCanvas().style.cursor = 'pointer';
    });
    map.current.on('mouseleave', 'unclustered-trees', () => {
      map.current.getCanvas().style.cursor = '';
    });

    return () => {
      map.current.off('click', 'tree-clusters', handleClusterClick);
      map.current.off('click', 'unclustered-trees', handleTreeClick);
      map.current.off('mouseenter', 'tree-clusters');
      map.current.off('mouseleave', 'tree-clusters');
      map.current.off('mouseenter', 'unclustered-trees');
      map.current.off('mouseleave', 'unclustered-trees');
    };
  }, []);

  return (
    <div className="map-container">
      <div ref={mapContainer} className="map" />
      <style jsx>{`
        .map-container {
          position: relative;
          height: 100vh;
          width: 100%;
        }
        .map {
          position: absolute;
          top: 0;
          bottom: 0;
          left: 0;
          right: 0;
        }
      `}</style>
    </div>
  );
};

export default Map;
