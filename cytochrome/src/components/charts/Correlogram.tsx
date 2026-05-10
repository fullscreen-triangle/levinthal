import { MARGIN } from "./constants";
import { Histogram } from "./Histogram";
import { Scatterplot } from "./Scatterplot";

type CorrelogramProps = {
  width: number;
  height: number;
  data: {
    var1: number;
    var2: number;
    var3: number;
    var4: number;
    group: "setosa" | "virginica" | "versicolor";
  }[];
};

export const Correlogram = ({ width, height, data }: CorrelogramProps) => {
  const boundsWidth = width - MARGIN.right - MARGIN.left;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;

  const allVariables = ["var1", "var2", "var3", "var4"] as const; // TODO: should not be hard-coded

  const allGroups = [...new Set(data.map((d) => d.group))];

  const graphWidth = boundsWidth / allVariables.length;
  const graphHeight = boundsHeight / allVariables.length;

  const allGraphs = allVariables.map((yVar, i) => {
    return allVariables.map((xVar, j) => {
      // If x and y variables are the same (diagonal), use a distribution instead.
      if (xVar === yVar) {
        const distributionData = allGroups.map((group) => {
          return {
            group,
            values: data.filter((d) => d.group === group).map((d) => d[xVar]),
          };
        });

        return (
          <Histogram
            key={i + "-" + j}
            width={graphWidth}
            height={graphHeight}
            data={distributionData}
            limits={[0, 8]}
          />
        );
      }

      // Scatterplot dataset
      const scatterData = data.map((d) => {
        return { x: d[xVar], y: d[yVar], group: d.group };
      });

      return (
        <div key={i + "-" + j}>
          <Scatterplot
            width={graphWidth}
            height={graphHeight}
            data={scatterData}
            yLabel={j === 0 ? allVariables[i] : undefined}
            xLabel={i === allVariables.length - 1 ? allVariables[j] : undefined}
          />
        </div>
      );
    });
  });

  return (
    <div
      style={{
        width,
        height,
      }}
    >
      <div
        style={{
          width: boundsWidth,
          height: boundsHeight,
          display: "grid",
          gridTemplateColumns: "1fr ".repeat(allVariables.length),
          transform: `translate(${MARGIN.left}px, ${MARGIN.top}px)`,
        }}
      >
        {allGraphs}
      </div>
    </div>
  );
};

import { useMemo } from "react";
import * as d3 from "d3";
import { colors, MARGIN } from "./constants";
import { AxisBottom } from "./Axis/AxisBottom";

const BUCKET_NUMBER = 40;
const BUCKET_PADDING = 0;

type HistogramProps = {
  width: number;
  height: number;
  data: { group: string; values: number[] }[];
  limits: [number, number];
};

export const Histogram = ({ width, height, data, limits }: HistogramProps) => {
  const boundsWidth = width - MARGIN.right - MARGIN.left;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;

  const allGroups = [...new Set(data.map((d) => d.group))].sort();
  const colorScale = d3.scaleOrdinal<string>().domain(allGroups).range(colors);

  const xScale = useMemo(() => {
    return d3.scaleLinear().domain(limits).range([10, boundsWidth]).nice();
  }, [data, width]);

  const bucketGenerator = useMemo(() => {
    return d3
      .bin()
      .value((d) => d)
      .domain(xScale.domain())
      .thresholds(xScale.ticks(BUCKET_NUMBER));
  }, [xScale]);

  const groupBuckets = useMemo(() => {
    return data.map((group) => {
      return { group, buckets: bucketGenerator(group.values) };
    });
  }, [data]);

  const yScale = useMemo(() => {
    const max = Math.max(
      ...groupBuckets.map((group) =>
        Math.max(...group.buckets.map((bucket) => bucket?.length))
      )
    );
    return d3.scaleLinear().range([boundsHeight, 0]).domain([0, max]).nice();
  }, [data, height]);

  const allRects = groupBuckets.map((group, i) =>
    group.buckets.map((bucket, j) => (
      <rect
        key={i + "_" + j}
        x={xScale(bucket.x0) + BUCKET_PADDING / 2}
        width={xScale(bucket.x1) - xScale(bucket.x0) - BUCKET_PADDING}
        y={yScale(bucket.length)}
        height={boundsHeight - yScale(bucket.length)}
        fill={colorScale(group.group.group)}
        opacity={1}
      />
    ))
  );

  return (
    <svg width={width} height={height}>
      <g
        width={boundsWidth}
        height={boundsHeight}
        transform={`translate(${[MARGIN.left, MARGIN.top].join(",")})`}
      >
        {allRects}

        {/* X axis, use an additional translation to appear at the bottom */}
        <g transform={`translate(0, ${boundsHeight})`}>
          <AxisBottom xScale={xScale} pixelsPerTick={20} />
        </g>
      </g>
    </svg>
  );
};
import { useMemo, useRef } from "react";
import * as d3 from "d3";
import { colors, MARGIN } from "./constants";
import { AxisLeft } from "./Axis/AxisLeft";
import { AxisBottom } from "./Axis/AxisBottom";

type ScatterplotProps = {
  width: number;
  height: number;
  data: { y: number; x: number; group: string }[];
  xLabel?: string;
  yLabel?: string;
};

export const Scatterplot = ({
  width,
  height,
  data,
  xLabel,
  yLabel,
}: ScatterplotProps) => {
  // Layout. The div size is set by the given props.
  // The bounds (=area inside the axis) is calculated by substracting the margins
  const boundsWidth = width - MARGIN.right - MARGIN.left;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;

  // Y axis
  const yScale = useMemo(() => {
    const [min, max] = d3.extent(data.map((d) => d.y));
    return d3.scaleLinear().domain([min, max]).range([boundsHeight, 0]).nice();
  }, [data, height]);

  // Y axis
  const xScale = useMemo(() => {
    const [min, max] = d3.extent(data.map((d) => d.x));
    return d3.scaleLinear().domain([0, max]).range([0, boundsWidth]).nice();
  }, [data, width]);

  // Color Scale
  const allGroups = [...new Set(data.map((d) => d.group))].sort();
  const colorScale = d3.scaleOrdinal<string>().domain(allGroups).range(colors);

  // Build the shapes
  const allShapes = data.map((d, i) => {
    return (
      <circle
        key={i}
        r={3}
        cx={xScale(d.x)}
        cy={yScale(d.y)}
        opacity={1}
        stroke={colorScale(d.group)}
        fill={colorScale(d.group)}
        fillOpacity={0.8}
        strokeWidth={1}
      />
    );
  });

  return (
    <svg width={width} height={height} style={{ overflow: "visible" }}>
      <g
        width={boundsWidth}
        height={boundsHeight}
        transform={`translate(${[MARGIN.left, MARGIN.top].join(",")})`}
      >
        {allShapes}

        {/* Y axis */}
        <AxisLeft yScale={yScale} pixelsPerTick={20} label={yLabel} />

        {/* X axis, use an additional translation to appear at the bottom */}
        <g transform={`translate(0, ${boundsHeight})`}>
          <AxisBottom xScale={xScale} pixelsPerTick={20} label={xLabel} />
        </g>
      </g>
    </svg>
  );
};
export const colors = [
    "#e0ac2b",
    "#e85252",
    "#6689c6",
    "#9a6fb0",
    "#a53253",
    "#69b3a2",
  ];

export const MARGIN = { top: 20, right: 20, bottom: 20, left: 20 };

