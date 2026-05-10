import { useState } from "react";
import { Renderer } from "./Renderer";
import { Tooltip } from "./Tooltip";
import { COLOR_LEGEND_HEIGHT } from "./constants";
import { ColorLegend } from "./ColorLegend";
import * as d3 from "d3";
import { COLORS, THRESHOLDS } from "./constants";

type HeatmapProps = {
  width: number;
  height: number;
  data: { x: number; y: string; value: number | null }[];
};

export type InteractionData = {
  xLabel: string;
  yLabel: string;
  xPos: number;
  yPos: number;
  value: number | null;
};

export const Heatmap = ({ width, height, data }: HeatmapProps) => {
  const [hoveredCell, setHoveredCell] = useState<InteractionData | null>(null);

  // Color scale is computed here bc it must be passed to both the renderer and the legend
  const values = data
    .map((d) => d.value)
    .filter((d): d is number => d !== null);
  const max = d3.max(values) || 0;

  const colorScale = d3
    .scaleLinear<string>()
    .domain(THRESHOLDS.map((t) => t * max))
    .range(COLORS);

  return (
    <div style={{ position: "relative" }}>
      <Renderer
        width={width}
        height={height - COLOR_LEGEND_HEIGHT}
        data={data}
        setHoveredCell={setHoveredCell}
        colorScale={colorScale}
      />
      <Tooltip
        interactionData={hoveredCell}
        width={width}
        height={height - COLOR_LEGEND_HEIGHT}
      />
      <div style={{ width: "100%", display: "flex", justifyContent: "center" }}>
        <ColorLegend
          height={COLOR_LEGEND_HEIGHT}
          width={200}
          colorScale={colorScale}
          interactionData={hoveredCell}
        />
      </div>
    </div>
  );
};

import { InteractionData } from "./Heatmap";
import * as d3 from "d3";
import { useEffect, useRef } from "react";

type ColorLegendProps = {
  height: number;
  width: number;
  colorScale: d3.ScaleLinear<string, string, never>;
  interactionData: InteractionData | null;
};

const COLOR_LEGEND_MARGIN = { top: 0, right: 0, bottom: 50, left: 0 };

export const ColorLegend = ({
  height,
  colorScale,
  width,
  interactionData,
}: ColorLegendProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const boundsWidth =
    width - COLOR_LEGEND_MARGIN.right - COLOR_LEGEND_MARGIN.left;
  const boundsHeight =
    height - COLOR_LEGEND_MARGIN.top - COLOR_LEGEND_MARGIN.bottom;

  const domain = colorScale.domain();
  const max = domain[domain.length - 1];
  const xScale = d3.scaleLinear().range([0, boundsWidth]).domain([0, max]);

  const allTicks = xScale.ticks(4).map((tick) => {
    return (
      <g key={tick}>
        <line
          x1={xScale(tick)}
          x2={xScale(tick)}
          y1={0}
          y2={boundsHeight + 10}
          stroke="black"
        />
        <text
          x={xScale(tick)}
          y={boundsHeight + 20}
          fontSize={9}
          textAnchor="middle"
        >
          {tick / 1000 + "k"}
        </text>
      </g>
    );
  });

  const hoveredValue = interactionData?.value;
  const x = hoveredValue ? xScale(hoveredValue) : null;
  const triangleWidth = 9;
  const triangleHeight = 6;
  const triangle = x ? (
    <polygon
      points={`${x},0 ${x - triangleWidth / 2},${-triangleHeight} ${
        x + triangleWidth / 2
      },${-triangleHeight}`}
      fill="grey"
    />
  ) : null;

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");

    if (!context) {
      return;
    }

    for (let i = 0; i < boundsWidth; ++i) {
      context.fillStyle = colorScale((max * i) / boundsWidth);
      context.fillRect(i, 0, 1, boundsHeight);
    }
  }, [width, height]);

  return (
    <div style={{ width, height }}>
      <div
        style={{
          position: "relative",
          transform: `translate(${COLOR_LEGEND_MARGIN.left}px,
            ${COLOR_LEGEND_MARGIN.top}px`,
        }}
      >
        <canvas ref={canvasRef} width={boundsWidth} height={boundsHeight} />
        <svg
          width={boundsWidth}
          height={boundsHeight}
          style={{ position: "absolute", top: 0, left: 0, overflow: "visible" }}
        >
          {allTicks}
          {triangle}
        </svg>
      </div>
    </div>
  );
};
import { useMemo } from "react";
import * as d3 from "d3";
import { InteractionData } from "./Heatmap";
import { MARGIN } from "./constants";
import styles from "./renderer.module.css";
import { Dataset } from "./data";

type RendererProps = {
  width: number;
  height: number;
  data: Dataset;
  setHoveredCell: (hoveredCell: InteractionData | null) => void;
  colorScale: d3.ScaleLinear<string, string, never>;
};

export const Renderer = ({
  width,
  height,
  data,
  setHoveredCell,
  colorScale,
}: RendererProps) => {
  // bounds = area inside the axis
  const boundsWidth = width - MARGIN.right - MARGIN.left;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;

  const allYGroups = useMemo(() => [...new Set(data.map((d) => d.y))], [data]);
  const allXGroups = useMemo(
    () => [...new Set(data.map((d) => String(d.x)))],
    [data]
  );

  const xScale = useMemo(() => {
    return d3
      .scaleBand()
      .range([0, boundsWidth])
      .domain(allXGroups)
      .padding(0.1);
  }, [data, width]);

  const yScale = useMemo(() => {
    return d3
      .scaleBand<string>()
      .range([0, boundsHeight])
      .domain(allYGroups)
      .padding(0.1);
  }, [data, height]);

  const allRects = data.map((d, i) => {
    const xPos = xScale(String(d.x));
    const yPos = yScale(d.y);

    if (d.value === null || !xPos || !yPos) {
      return;
    }

    return (
      <rect
        key={i}
        x={xPos}
        y={yPos}
        className={styles.rectangle}
        width={xScale.bandwidth()}
        height={yScale.bandwidth()}
        fill={d.value ? colorScale(d.value) : "#F8F8F8"}
        onMouseEnter={(e) => {
          setHoveredCell({
            xLabel: String(d.x),
            yLabel: d.y,
            xPos: xPos + xScale.bandwidth() + MARGIN.left,
            yPos: yPos + xScale.bandwidth() / 2 + MARGIN.top,
            value: d.value ? Math.round(d.value * 100) / 100 : null,
          });
        }}
      />
    );
  });

  const xLabels = allXGroups.map((name, i) => {
    if (name && Number(name) % 10 === 0) {
      return (
        <text
          key={i}
          x={xScale(name)}
          y={boundsHeight + 10}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={10}
          stroke="none"
          fill="black"
        >
          {name}
        </text>
      );
    }
  });

  const yLabels = allYGroups.map((name, i) => {
    const yPos = yScale(name);
    if (yPos && i % 2 === 0) {
      return (
        <text
          key={i}
          x={-5}
          y={yPos + yScale.bandwidth() / 2}
          textAnchor="end"
          dominantBaseline="middle"
          fontSize={10}
        >
          {name}
        </text>
      );
    }
  });

  return (
    <svg
      width={width}
      height={height}
      onMouseLeave={() => setHoveredCell(null)}
    >
      <g
        width={boundsWidth}
        height={boundsHeight}
        transform={`translate(${[MARGIN.left, MARGIN.top].join(",")})`}
      >
        {allRects}
        {xLabels}
        {yLabels}
      </g>
    </svg>
  );
};
import { InteractionData } from "./Heatmap";
import styles from "./tooltip.module.css";

type TooltipProps = {
  interactionData: InteractionData | null;
  width: number;
  height: number;
};

export const Tooltip = ({ interactionData, width, height }: TooltipProps) => {
  if (!interactionData) {
    return null;
  }

  return (
    // Wrapper div: a rect on top of the viz area
    <div
      style={{
        width,
        height,
        position: "absolute",
        top: 0,
        left: 0,
        pointerEvents: "none",
      }}
    >
      {/* The actual box with white background */}
      <div
        className={styles.tooltip}
        style={{
          position: "absolute",
          left: interactionData.xPos,
          top: interactionData.yPos,
        }}
      >
        <span>{interactionData.yLabel}</span>
        <br />
        <span>{interactionData.xLabel}</span>
        <span>: </span>
        <b>{interactionData.value}</b>
      </div>
    </div>
  );
};
export const MARGIN = { top: 10, right: 10, bottom: 50, left: 50 };

export const COLOR_LEGEND_HEIGHT = 60;

export const COLORS = [
  "#e7f0fa",
  "#c9e2f6",
  "#95cbee",
  "#0099dc",
  "#4ab04a",
  "#ffd73e",
  "#eec73a",
  "#e29421",
  "#e29421",
  "#f05336",
  "#ce472e",
];

export const THRESHOLDS = [
  0, 0.01, 0.02, 0.03, 0.09, 0.1, 0.15, 0.25, 0.4, 0.5, 1,
];
