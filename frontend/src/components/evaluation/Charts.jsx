import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { CHART_COLORS } from '../../utils/constants';
import { strategyColor } from '../../utils/strategy';

/**
 * Shared chart primitives.
 *
 * A single axis/grid/tooltip treatment is applied everywhere so the Dashboard
 * and Evaluation pages read as one system. Series colours come from the
 * strategy palette wherever the series *is* a strategy, so a strategy keeps its
 * colour across every view.
 */

const AXIS = {
  stroke: '#a8b8cc',
  fontSize: 11,
  tickLine: false,
  axisLine: false,
};

const GRID = { stroke: '#e9edf3', strokeDasharray: '3 3', vertical: false };

function ChartTooltip({ active, payload, label, formatter }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-ink-200 bg-white px-3 py-2 shadow-panel">
      {label ? <p className="mb-1 text-[11px] font-medium text-ink-900">{label}</p> : null}
      {payload.map((entry) => (
        <p key={entry.dataKey || entry.name} className="flex items-center gap-1.5 text-[11px] text-ink-600">
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: entry.color || entry.payload?.fill }}
            aria-hidden="true"
          />
          <span className="capitalize">{entry.name}:</span>
          <span className="font-medium tabular-nums text-ink-900">
            {formatter ? formatter(entry.value) : entry.value}
          </span>
        </p>
      ))}
    </div>
  );
}

export function EmptyChart({ message = 'No data yet' }) {
  return (
    <div className="flex h-full min-h-[200px] items-center justify-center rounded-lg border border-dashed border-ink-200 bg-ink-50/40">
      <p className="px-4 text-center text-xs text-ink-400">{message}</p>
    </div>
  );
}

/** Queries or latency over time. */
export function TimeseriesChart({ data = [], dataKey = 'value', color = '#3567f0', height = 220, formatter, label }) {
  if (!data.length) return <EmptyChart message="No activity in this period yet." />;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="date" {...AXIS} minTickGap={24} />
        <YAxis {...AXIS} width={44} allowDecimals={false} />
        <Tooltip content={<ChartTooltip formatter={formatter} />} cursor={{ stroke: '#cfd8e3' }} />
        <Line
          type="monotone"
          dataKey={dataKey}
          name={label || dataKey}
          stroke={color}
          strokeWidth={2}
          dot={{ r: 2.5, fill: color, strokeWidth: 0 }}
          activeDot={{ r: 4 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Distribution bars — strategy usage, complexity, intent, confidence. */
export function DistributionChart({
  data = [],
  height = 220,
  useStrategyColors = false,
  horizontal = false,
  formatter,
}) {
  if (!data.length) return <EmptyChart message="No distribution data yet." />;

  const bars = data.map((item, index) => ({
    ...item,
    fill: useStrategyColors ? strategyColor(item.label) : CHART_COLORS[index % CHART_COLORS.length],
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={bars}
        layout={horizontal ? 'vertical' : 'horizontal'}
        margin={{ top: 8, right: 12, bottom: 0, left: horizontal ? 8 : -12 }}
      >
        <CartesianGrid {...GRID} vertical={horizontal} horizontal={!horizontal} />
        {horizontal ? (
          <>
            <XAxis type="number" {...AXIS} allowDecimals={false} />
            <YAxis type="category" dataKey="label" {...AXIS} width={96} />
          </>
        ) : (
          <>
            <XAxis dataKey="label" {...AXIS} interval={0} angle={data.length > 5 ? -20 : 0} textAnchor={data.length > 5 ? 'end' : 'middle'} height={data.length > 5 ? 48 : 30} />
            <YAxis {...AXIS} width={40} allowDecimals={false} />
          </>
        )}
        <Tooltip content={<ChartTooltip formatter={formatter} />} cursor={{ fill: '#f5f7fa' }} />
        <Bar dataKey="value" name="count" radius={horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]} maxBarSize={44}>
          {bars.map((item) => (
            <Cell key={item.label} fill={item.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Provider split. */
export function DonutChart({ data = [], height = 200, formatter }) {
  if (!data.length) return <EmptyChart message="No provider activity yet." />;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="label"
          innerRadius="55%"
          outerRadius="80%"
          paddingAngle={2}
          strokeWidth={0}
        >
          {data.map((item, index) => (
            <Cell key={item.label} fill={CHART_COLORS[index % CHART_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip content={<ChartTooltip formatter={formatter} />} />
        <Legend
          verticalAlign="bottom"
          height={28}
          iconType="circle"
          iconSize={7}
          formatter={(value) => <span className="text-[11px] capitalize text-ink-600">{value}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}

/**
 * Grouped comparison of one metric across strategies.
 * Used on the Evaluation page for side-by-side benchmark results.
 */
export function StrategyComparisonChart({ data = [], metricKey, metricLabel, height = 260, formatter }) {
  if (!data.length) return <EmptyChart message="Run an evaluation to populate this chart." />;

  const rows = data
    .filter((row) => row[metricKey] !== null && row[metricKey] !== undefined)
    .map((row) => ({
      strategy: row.strategy,
      value: row[metricKey],
      fill: strategyColor(row.strategy),
    }));

  if (!rows.length) {
    return <EmptyChart message={`${metricLabel} was not computed for these runs.`} />;
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="strategy" {...AXIS} interval={0} angle={-20} textAnchor="end" height={52} />
        <YAxis {...AXIS} width={52} />
        <Tooltip content={<ChartTooltip formatter={formatter} />} cursor={{ fill: '#f5f7fa' }} />
        <Bar dataKey="value" name={metricLabel} radius={[4, 4, 0, 0]} maxBarSize={48}>
          {rows.map((row) => (
            <Cell key={row.strategy} fill={row.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Multi-metric profile per strategy — shows trade-offs, not just winners. */
export function MetricRadarChart({ runs = [], metrics = [], height = 300 }) {
  if (!runs.length || !metrics.length) {
    return <EmptyChart message="Run at least one evaluation to compare metric profiles." />;
  }

  const data = metrics.map((metric) => {
    const point = { metric: metric.label };
    runs.forEach((run) => {
      const value = run[metric.key];
      point[run.strategy] = value === null || value === undefined ? 0 : value;
    });
    return point;
  });

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RadarChart data={data} outerRadius="72%">
        <PolarGrid stroke="#e9edf3" />
        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: '#547296' }} />
        <PolarRadiusAxis domain={[0, 1]} tick={{ fontSize: 9, fill: '#a8b8cc' }} axisLine={false} />
        {runs.map((run) => (
          <Radar
            key={run.strategy}
            name={run.strategy}
            dataKey={run.strategy}
            stroke={strategyColor(run.strategy)}
            fill={strategyColor(run.strategy)}
            fillOpacity={0.12}
            strokeWidth={2}
          />
        ))}
        <Legend
          iconType="circle"
          iconSize={7}
          formatter={(value) => <span className="text-[11px] capitalize text-ink-600">{value}</span>}
        />
        <Tooltip content={<ChartTooltip formatter={(v) => Number(v).toFixed(3)} />} />
      </RadarChart>
    </ResponsiveContainer>
  );
}
