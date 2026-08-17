import { useCallback, useEffect, useMemo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  Handle,
  MarkerType,
  Position,
  useEdgesState,
  useNodesState,
  useReactFlow,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { ENTITY_TYPE_COLORS } from '../../utils/constants';
import cn from '../../utils/cn';

/** Entity node. Size scales with degree so hubs are visually obvious. */
function EntityNode({ data, selected }) {
  const color = ENTITY_TYPE_COLORS[data.type] || ENTITY_TYPE_COLORS.CONCEPT;
  const isHub = (data.degree || 0) >= 4;

  return (
    <div
      className={cn(
        'rounded-lg border-2 bg-white px-3 py-2 shadow-card transition',
        selected ? 'ring-2 ring-brand-400 ring-offset-2' : '',
        data.dimmed ? 'opacity-25' : '',
        data.highlighted ? 'shadow-card-hover' : '',
      )}
      style={{ borderColor: color, minWidth: isHub ? 150 : 120, maxWidth: 210 }}
    >
      <Handle type="target" position={Position.Left} />
      <div className="flex items-center gap-1.5">
        <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: color }} aria-hidden="true" />
        <span
          className={cn('truncate font-medium text-ink-900', isHub ? 'text-[13px]' : 'text-xs')}
          title={data.label}
        >
          {data.label}
        </span>
      </div>
      <div className="mt-0.5 flex items-center justify-between gap-2">
        <span className="truncate text-[9px] uppercase tracking-wide" style={{ color }}>
          {data.type}
        </span>
        {data.degree ? (
          <span className="shrink-0 text-[9px] text-ink-400">{data.degree} links</span>
        ) : null}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const NODE_TYPES = { entity: EntityNode };

/**
 * Deterministic radial-ish layout.
 *
 * The graph has no positions of its own, so nodes are placed on concentric
 * rings ordered by degree: hubs land near the centre, leaves on the outside.
 * A seeded offset keeps the layout stable across renders of the same data.
 */
function layout(nodes) {
  const sorted = [...nodes].sort((a, b) => (b.degree || 0) - (a.degree || 0));
  const positions = new Map();

  const RING_SIZES = [1, 6, 12, 20, 30, 42];
  let index = 0;
  let ring = 0;

  while (index < sorted.length) {
    const capacity = RING_SIZES[Math.min(ring, RING_SIZES.length - 1)];
    const count = Math.min(capacity, sorted.length - index);
    const radius = ring === 0 ? 0 : 190 * ring;
    const offset = ring * 0.4;

    for (let i = 0; i < count; i += 1) {
      const node = sorted[index + i];
      const angle = (i / count) * Math.PI * 2 + offset;
      positions.set(node.id, {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius * 0.82,
      });
    }

    index += count;
    ring += 1;
  }

  return positions;
}

function FlowInner({ nodes: rawNodes, edges: rawEdges, selectedId, onSelect, searchTerm }) {
  const { fitView } = useReactFlow();

  const { flowNodes, flowEdges } = useMemo(() => {
    const positions = layout(rawNodes);
    const term = searchTerm?.trim().toLowerCase();

    // When a node is selected, dim everything not directly connected to it.
    const connected = new Set();
    if (selectedId) {
      connected.add(selectedId);
      rawEdges.forEach((edge) => {
        if (edge.source === selectedId) connected.add(edge.target);
        if (edge.target === selectedId) connected.add(edge.source);
      });
    }

    const builtNodes = rawNodes.map((node) => {
      const matchesSearch = term ? node.name.toLowerCase().includes(term) : false;
      const dimmed =
        (selectedId && !connected.has(node.id)) || (term && !matchesSearch);

      return {
        id: node.id,
        type: 'entity',
        position: positions.get(node.id) || { x: 0, y: 0 },
        data: {
          label: node.name,
          type: node.type,
          degree: node.degree,
          description: node.description,
          documents: node.documents,
          chunkIds: node.chunk_ids,
          dimmed,
          highlighted: matchesSearch,
        },
        selected: node.id === selectedId,
      };
    });

    const builtEdges = rawEdges.map((edge) => {
      const active = !selectedId || edge.source === selectedId || edge.target === selectedId;
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: edge.type?.replace(/_/g, ' ').toLowerCase(),
        type: 'smoothstep',
        animated: Boolean(selectedId) && active,
        style: {
          stroke: active ? '#7791b0' : '#e9edf3',
          strokeWidth: active ? 1.6 : 1,
          opacity: active ? 1 : 0.35,
        },
        labelStyle: {
          fontSize: 9,
          fill: active ? '#547296' : '#cfd8e3',
        },
        labelBgStyle: { fill: '#ffffff', fillOpacity: 0.9 },
        labelBgPadding: [3, 1],
        labelBgBorderRadius: 3,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 14,
          height: 14,
          color: active ? '#7791b0' : '#e9edf3',
        },
      };
    });

    return { flowNodes: builtNodes, flowEdges: builtEdges };
  }, [rawNodes, rawEdges, selectedId, searchTerm]);

  const [nodes, setNodes, onNodesChange] = useNodesState(flowNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(flowEdges);

  useEffect(() => {
    setNodes(flowNodes);
    setEdges(flowEdges);
  }, [flowNodes, flowEdges, setNodes, setEdges]);

  useEffect(() => {
    const timer = setTimeout(() => fitView({ padding: 0.18, duration: 400 }), 80);
    return () => clearTimeout(timer);
  }, [rawNodes.length, fitView]);

  const handleNodeClick = useCallback(
    (_event, node) => {
      onSelect?.(node.id === selectedId ? null : node.id);
    },
    [onSelect, selectedId],
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      nodeTypes={NODE_TYPES}
      onNodeClick={handleNodeClick}
      onPaneClick={() => onSelect?.(null)}
      minZoom={0.12}
      maxZoom={2}
      proOptions={{ hideAttribution: true }}
      fitView
      nodesDraggable
      nodesConnectable={false}
      elementsSelectable
    >
      <Background color="#cfd8e3" gap={22} size={1} />
      <Controls showInteractive={false} position="bottom-right" />
    </ReactFlow>
  );
}

export default function GraphCanvas(props) {
  return <FlowInner {...props} />;
}
