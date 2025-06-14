import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useEnvironmentalContext } from '../contexts/EnvironmentalContext';

const DataVisualization = () => {
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);
  const { environmentalMetrics } = useEnvironmentalContext();

  useEffect(() => {
    if (!environmentalMetrics || !environmentalMetrics.species_distribution) return;

    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Clear previous SVG
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Prepare data
    const data = environmentalMetrics.species_distribution
      .sort((a, b) => b.count - a.count)
      .slice(0, 10); // Top 10 species

    // Scales
    const x = d3.scaleBand()
      .range([0, width])
      .padding(0.1);

    const y = d3.scaleLinear()
      .range([height, 0]);

    x.domain(data.map(d => d.species));
    y.domain([0, d3.max(data, d => d.count)]);

    // Add X axis
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end');

    // Add Y axis
    svg.append('g')
      .call(d3.axisLeft(y));

    // Create tooltip
    const tooltip = d3.select(tooltipRef.current);

    // Add bars
    svg.selectAll('.bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', d => x(d.species))
      .attr('width', x.bandwidth())
      .attr('y', d => y(d.count))
      .attr('height', d => height - y(d.count))
      .attr('fill', '#69b3a2')
      .on('mouseover', (event, d) => {
        tooltip
          .style('opacity', 1)
          .html(`
            <strong>${d.species}</strong><br/>
            Count: ${d.count}<br/>
            Avg Height: ${d.avg_height.toFixed(2)}m<br/>
            Avg Diameter: ${d.avg_diameter.toFixed(2)}cm
          `)
          .style('left', `${event.pageX + 10}px`)
          .style('top', `${event.pageY - 28}px`);
      })
      .on('mouseout', () => {
        tooltip.style('opacity', 0);
      });

    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 0 - margin.top / 2)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .text('Top 10 Tree Species Distribution');

    // Add labels
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - height / 2)
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .text('Number of Trees');

  }, [environmentalMetrics]);

  return (
    <div className="visualization-container">
      <svg ref={svgRef}></svg>
      <div ref={tooltipRef} className="tooltip"></div>
      <style jsx>{`
        .visualization-container {
          background: white;
          border-radius: 8px;
          padding: 20px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          margin: 20px;
        }
        .tooltip {
          position: absolute;
          padding: 8px;
          background: rgba(255, 255, 255, 0.9);
          border: 1px solid #ddd;
          border-radius: 4px;
          pointer-events: none;
          opacity: 0;
          transition: opacity 0.2s;
        }
        :global(.bar:hover) {
          opacity: 0.8;
        }
      `}</style>
    </div>
  );
};

export default DataVisualization;
