# DAG Visualization

Interactive visualization of how Lossless Context Management builds and navigates the summary DAG.

<div style="position: relative; width: 100%; height: 0; padding-bottom: 75%; overflow: hidden; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.2);">
  <iframe src="visualization.html" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;" allowfullscreen></iframe>
</div>

!!! tip "Full Screen"
    For the best experience, [open the visualization in a new tab](visualization.html){target=_blank}.

## What You're Seeing

- **Blue circles** — User messages
- **Purple circles** — Assistant responses  
- **Amber circles** — Tool results
- **Green rounded rectangles** — Leaf summaries (depth 0)
- **Teal rounded rectangles** — Condensed summaries (depth 1+)

## How to Interact

1. **Click** any summary node to expand/collapse its children
2. **Hover** over nodes for content preview
3. Use **Animate Compaction Demo** to watch the DAG build in real-time
4. Use **Step Through Build** to add nodes one at a time
5. **Zoom** with scroll wheel, **pan** by dragging the background
6. **Drag** individual nodes to rearrange
