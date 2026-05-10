import * as d3 from 'd3';
import { Data } from './data';
import { AxisConfig, INNER_RADIUS, RadarGrid } from './RadarGrid';

const MARGIN = 30;
const COLORS = [
  '#e0ac2b',
  '#e85252',
  '#6689c6',
  '#9a6fb0',
  '#a53253',
  '#69b3a2',
];

type YScale = d3.ScaleRadial<number, number, never>;

type RadarProps = {
  width: number;
  height: number;
  data: Data;
  axisConfig: AxisConfig[];
};

/*
  A react component that builds a Radar Chart for several groups in the dataset
*/
export const Radar = ({ width, height, data, axisConfig }: RadarProps) => {
  const outerRadius = Math.min(width, height) / 2 - MARGIN;

  // The x scale provides an angle for each variable of the dataset
  const allVariableNames = axisConfig.map((axis) => axis.name);
  const xScale = d3
    .scaleBand()
    .domain(allVariableNames)
    .range([0, 2 * Math.PI]);

  // Compute the y scales: 1 scale per variable.
  // Provides the distance to the center.
  let yScales: { [name: string]: YScale } = {};
  axisConfig.forEach((axis) => {
    yScales[axis.name] = d3
      .scaleRadial()
      .domain([0, axis.max])
      .range([INNER_RADIUS, outerRadius]);
  });

  // Color Scale
  const allGroups = data.map((d) => d.name);
  const colorScale = d3.scaleOrdinal<string>().domain(allGroups).range(COLORS);

  // Compute the main radar shapes, 1 per group
  const lineGenerator = d3.lineRadial();

  const allLines = data.map((series, i) => {
    const allCoordinates = axisConfig.map((axis) => {
      const yScale = yScales[axis.name];
      const angle = xScale(axis.name) ?? 0; // I don't understand the type of scalePoint. IMO x cannot be undefined since I'm passing it something of type Variable.
      const radius = yScale(series[axis.name]);
      const coordinate: [number, number] = [angle, radius];
      return coordinate;
    });

    // To close the path of each group, the path must finish where it started
    // so add the last data point at the end of the array
    allCoordinates.push(allCoordinates[0]);

    const d = lineGenerator(allCoordinates);

    if (!d) {
      return;
    }

    return (
      <path
        key={i}
        d={d}
        stroke={colorScale(series.name)}
        strokeWidth={3}
        fill={colorScale(series.name)}
        fillOpacity={0.1}
      />
    );
  });

  return (
    <svg width={width} height={height}>
      <g transform={'translate(' + width / 2 + ',' + height / 2 + ')'}>
        <RadarGrid
          outerRadius={outerRadius}
          xScale={xScale}
          axisConfig={axisConfig}
        />
        {allLines}
      </g>
    </svg>
  );
};
import { Variable } from './data';
import { polarToCartesian } from './polarToCartesian';
import * as d3 from 'd3';

//
// Constants
//
export const INNER_RADIUS = 40;
const GRID_NUMBER = 5;
const GRID_COLOR = 'lightGrey';

//
// Types
//
export type AxisConfig = {
  name: Variable;
  max: number;
};

type RadarGridProps = {
  outerRadius: number;
  xScale: d3.ScaleBand<string>;
  axisConfig: AxisConfig[];
};

/*
  A react component that adds a grid background 
  for a radar chart in polar coordinates
*/
export const RadarGrid = ({
  outerRadius,
  xScale,
  axisConfig,
}: RadarGridProps) => {
  const lineGenerator = d3.lineRadial();

  // Compute Axes = from center to outer
  const allAxes = axisConfig.map((axis, i) => {
    const angle = xScale(axis.name);

    if (angle === undefined) {
      return null;
    }

    const path = lineGenerator([
      [angle, INNER_RADIUS],
      [angle, outerRadius],
    ]);

    const labelPosition = polarToCartesian(
      angle - Math.PI / 2,
      outerRadius + 10
    );

    return (
      <g key={i}>
        <path d={path} stroke={GRID_COLOR} strokeWidth={0.5} rx={1} />
        <text
          x={labelPosition.x}
          y={labelPosition.y}
          fontSize={12}
          fill={GRID_COLOR}
          textAnchor={labelPosition.x > 0 ? 'start' : 'end'}
          dominantBaseline="middle"
        >
          {axis.name}
        </text>
      </g>
    );
  });

  // Compte grid = concentric circles
  const allCircles = [...Array(GRID_NUMBER).keys()].map((position, i) => {
    return (
      <circle
        key={i}
        cx={0}
        cy={0}
        r={
          INNER_RADIUS +
          (position * (outerRadius - INNER_RADIUS)) / (GRID_NUMBER - 1)
        }
        stroke={GRID_COLOR}
        fill="none"
      />
    );
  });

  return (
    <g>
      {allAxes}
      {allCircles}
    </g>
  );
};
