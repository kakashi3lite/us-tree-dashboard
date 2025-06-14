import React, { useMemo } from 'react';
import { useQuery } from 'react-query';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts';
import { Card, CardContent, Typography, Grid, useMediaQuery } from '@mui/material';
import { format } from 'date-fns';
import { fetchHistoricalTrends, fetchEnvironmentalImpact } from '../services/api';

const DataVisualization = ({ regionId, startDate, endDate }) => {
  const isMobile = useMediaQuery('(max-width:600px)');

  const { data: trends = {} } = useQuery(
    ['historicalTrends', regionId, startDate, endDate],
    () => fetchHistoricalTrends('height', 'month', startDate, endDate),
    {
      keepPreviousData: true,
      staleTime: 300000
    }
  );

  const { data: impact = {} } = useQuery(
    ['environmentalImpact', regionId, startDate, endDate],
    () => fetchEnvironmentalImpact(regionId, startDate, endDate),
    {
      keepPreviousData: true,
      staleTime: 300000
    }
  );

  const formattedTrends = useMemo(() => {
    return trends.map(point => ({
      ...point,
      period: format(new Date(point.period), 'MMM yyyy'),
      value: Number(point.value).toFixed(2),
      changePercent: Number(point.change_percent).toFixed(1)
    }));
  }, [trends]);

  const impactMetrics = useMemo(() => [
    {
      label: 'CO₂ Absorbed',
      value: `${Number(impact.total_co2_absorbed || 0).toFixed(1)} tons`,
      color: '#2196f3'
    },
    {
      label: 'Oxygen Produced',
      value: `${Number(impact.total_oxygen_produced || 0).toFixed(1)} tons`,
      color: '#4caf50'
    },
    {
      label: 'Water Filtered',
      value: `${Number(impact.total_water_filtered || 0).toFixed(1)} gallons`,
      color: '#03a9f4'
    }
  ], [impact]);

  const chartHeight = isMobile ? 200 : 300;

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Tree Growth Trends
            </Typography>
            <ResponsiveContainer width="100%" height={chartHeight}>
              <LineChart data={formattedTrends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="period"
                  angle={-45}
                  textAnchor="end"
                  height={60}
                  interval={isMobile ? 1 : 0}
                />
                <YAxis />
                <Tooltip
                  formatter={(value, name) => [
                    `${value} ft`,
                    name === 'value' ? 'Height' : 'Change'
                  ]}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#2196f3"
                  strokeWidth={2}
                  dot={false}
                  name="Height"
                />
                <Line
                  type="monotone"
                  dataKey="changePercent"
                  stroke="#4caf50"
                  strokeWidth={2}
                  dot={false}
                  name="Change %"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Environmental Impact
            </Typography>
            <ResponsiveContainer width="100%" height={chartHeight}>
              <BarChart data={impactMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="label"
                  angle={-45}
                  textAnchor="end"
                  height={60}
                  interval={0}
                />
                <YAxis />
                <Tooltip
                  formatter={(value, name, props) => [
                    props.payload.value,
                    props.payload.label
                  ]}
                />
                <Bar
                  dataKey="value"
                  fill="#2196f3"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default DataVisualization;