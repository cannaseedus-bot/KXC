# GUIX Hybrid Model - API Reference

## Overview

The GUIX Hybrid Model is a Phase 7 intelligent multi-format rendering system that auto-detects technology stacks (SVG, CSS, HTML) and provides format-agnostic component rendering. It combines three core modules:

- **GUIXStackAnalyzer**: Detects technology stacks with confidence scoring
- **GUIXVisualComponents**: 30+ pre-built reusable components
- **GUIXRenderer**: Multi-format rendering engine with format conversion
- **GUIXHybridModel**: Main orchestrator with theme management and tensor optimization

## Table of Contents

1. [GUIXHybridModel](#guixhybridmodel-api)
2. [GUIXStackAnalyzer](#guixstackanalyzer-api)
3. [GUIXVisualComponents](#guixvisualcomponents-api)
4. [GUIXRenderer](#guixrenderer-api)
5. [Theming System](#theming-system)
6. [Component Library](#component-library)

---

## GUIXHybridModel API

The main orchestration class for the entire GUIX system.

### Constructor

```javascript
import { GUIXHybridModel } from './src/guix-hybrid-model.js';

const guix = new GUIXHybridModel(config);
```

**Config Options:**
- `cacheEnabled` (boolean): Enable component caching (default: true)
- `maxCacheSize` (number): Maximum cache size in items (default: 10000)
- `defaultTheme` (string): Default theme name (default: 'light')
- `tensorOptimization` (boolean): Enable tensor-based optimization (default: true)
- `autoConvert` (boolean): Auto-convert between formats (default: true)

### Methods

#### render(content, options)

Detect technology stack and render content.

```javascript
const result = guix.render('<svg><circle/></svg>', {
  targetFormat: 'html',    // null = auto-detect
  theme: 'light',
  optimize: false,
  minify: false,
  autoDetect: true
});

// Returns:
// {
//   output: string,
//   format: string,
//   detectedFormat: string,
//   stackAnalysis: object,
//   themeApplied: string,
//   timestamp: number
// }
```

#### renderComponent(componentName, data, options)

Create and render a pre-built component.

```javascript
const result = guix.renderComponent('svgLineChart', {
  title: 'Sales Data',
  points: [[0, 10], [1, 20], [2, 15]]
}, {
  theme: 'dark',
  targetFormat: null,      // null = use component's native format
  optimize: false,
  minify: false
});
```

#### convertFormat(content, fromFormat, toFormat, options)

Convert between formats.

```javascript
const result = guix.convertFormat(
  '<svg></svg>',           // content
  'svg',                   // fromFormat (null = auto-detect)
  'html',                  // toFormat
  {}                       // options
);

// Returns: { output, fromFormat, toFormat, format, converted, timestamp }
```

#### autoOptimize(content)

Auto-detect and provide optimization recommendations.

```javascript
const optimization = guix.autoOptimize('<div>Content</div>');

// Returns:
// {
//   detectedFormat: 'html',
//   confidenceScores: { svg: 0, css: 0, html: 1 },
//   optimization: { ... },
//   suggestions: [ ... ],
//   hybrid: false,
//   conversionPaths: { ... }
// }
```

#### Theme Management

**setTheme(themeName)**
```javascript
guix.setTheme('dark');
```

**getCurrentTheme()**
```javascript
const theme = guix.getCurrentTheme();
// { name, colors, typography }
```

**getThemes()**
```javascript
const themes = guix.getThemes();
// ['light', 'dark', 'custom']
```

**createTheme(themeName, themeConfig)**
```javascript
guix.createTheme('myTheme', {
  colors: {
    primary: '#ff0000',
    secondary: '#00ff00',
    // ... other colors
  },
  typography: {
    fontFamily: 'Arial',
    fontSize: '16px'
  }
});
```

#### Component Discovery

**getComponents()**
```javascript
const registry = guix.getComponents();
// Returns all registered components with metadata
```

**listComponents(category)**
```javascript
const htmlComponents = guix.listComponents('content');
// Returns filtered component array
```

**getComponentsByFormat(format)**
```javascript
const svgComponents = guix.getComponentsByFormat('svg');
```

#### Accessibility & Documentation

**getComponentAccessibility(componentName)**
```javascript
const a11y = guix.getComponentAccessibility('htmlForm');
// {
//   componentName: 'htmlForm',
//   accessibility: ['semantic-markup', 'aria-labels', ...],
//   wcagLevel: 'AA',
//   features: { semanticMarkup: true, ariaLabels: true, ... }
// }
```

**getComponentDocumentation(componentName)**
```javascript
const docs = guix.getComponentDocumentation('svgLineChart');
// { name, format, category, accessibility, features, recommendations }
```

#### Performance & Metrics

**getTensorMetrics()**
```javascript
const metrics = guix.getTensorMetrics();
// {
//   renderTime: { average, samples, min, max },
//   cache: { hits, misses, hitRate },
//   components: { created },
//   conversions: { performed }
// }
```

**getCacheStats()**
```javascript
const stats = guix.getCacheStats();
// { componentCacheSize, renderCacheSize, totalCacheSize }
```

**clearCaches()**
```javascript
guix.clearCaches();
```

**resetMetrics()**
```javascript
guix.resetMetrics();
```

#### Serialization

**serialize()**
```javascript
const state = guix.serialize();
// Returns JSON string of current state
```

**deserialize(serialized)**
```javascript
guix.deserialize(stateJson);
```

#### System Info

**getSystemInfo()**
```javascript
const info = guix.getSystemInfo();
// {
//   version: '7.0.0',
//   phase: 'Phase 7 - GUIX Hybrid Model',
//   components: 30,
//   themes: 3,
//   supportedFormats: ['svg', 'css', 'html', 'json']
// }
```

---

## GUIXStackAnalyzer API

Intelligent technology stack detection engine.

### Methods

#### analyzeStack(content)

Analyze content and detect technology stack.

```javascript
const analyzer = new GUIXStackAnalyzer();
const result = analyzer.analyzeStack('<svg><circle/></svg>');

// Returns:
// {
//   svg: { score, confidence, matches, isVector },
//   css: { score, confidence, matches, isStylesheet },
//   html: { score, confidence, matches, isSemantic },
//   primary: 'svg',
//   hybrid: false,
//   conversionPath: { ... },
//   timestamp: number
// }
```

#### detectSVG(content)
#### detectCSS(content)
#### detectHTML(content)

Individual format detection with detailed metrics.

```javascript
const svgResult = analyzer.detectSVG(content);
// { score, matches, keywordMatches, isVector }
```

#### isHybridContent(content)

Check if content uses multiple technology stacks.

```javascript
const isHybrid = analyzer.isHybridContent(htmlWithSvgAndCss);
// true if content spans multiple stacks
```

#### optimizeForStack(content, targetStack)

Get optimization recommendations for a stack.

```javascript
const optimizations = analyzer.optimizeForStack(content, 'svg');
// { svg: [...], css: [...], html: [...], appliedOptimizations: [...] }
```

#### getOptimizationHints(stack)

Get best practices for stack optimization.

```javascript
const hints = analyzer.getOptimizationHints('svg');
// {
//   minification: '...',
//   performance: '...',
//   accessibility: '...',
//   optimization: '...'
// }
```

#### Cache Management

**getCacheStats()**: Get cache usage statistics
**clearCache()**: Clear detection cache

---

## GUIXVisualComponents API

Pre-built component library with 30+ components.

### Categories

- **Visualization** (10): SVG charts and diagrams
- **Layout** (3): CSS layout patterns
- **Component** (7): CSS styled components
- **Semantic** (4): HTML semantic markup
- **Content** (6): HTML content structures

### SVG Components (Visualization)

#### Chart Components
- `svgLineChart`: Line chart with configurable points
- `svgBarChart`: Bar chart with categories
- `svgPieChart`: Pie/donut chart with slices
- `svgScatterPlot`: Scatter plot visualization
- `svgGauge`: Gauge/meter visualization
- `svgRadar`: Radar/spider chart

#### Diagram Components
- `svgHeatmap`: 2D heatmap matrix
- `svgNetwork`: Network graph with nodes and edges
- `svgHierarchy`: Hierarchical tree diagram
- `svgFlowchart`: Process flowchart with steps

### CSS Components (Layouts & Styles)

#### Layouts
- `cssGrid`: CSS Grid layout (configurable columns)
- `cssFlexbox`: Flexbox layout (configurable direction)
- `cssSidebar`: Sidebar + content layout

#### Styled Components
- `cssNavbar`: Navigation bar
- `cssButton`: Styled button (sizes: sm, md, lg)
- `cssCard`: Card with header/body/footer
- `cssModal`: Modal dialog overlay
- `cssAlert`: Alert messages (types: info, success, warning, danger)
- `cssTooltip`: Tooltip component
- `cssDropdown`: Dropdown menu

### HTML Components (Semantic)

#### Navigation
- `htmlNavigation`: Semantic nav element
- `htmlBreadcrumb`: Breadcrumb navigation
- `htmlTabs`: Tab interface

#### Content
- `htmlArticle`: Article with header and metadata
- `htmlAside`: Aside sidebar content
- `htmlFooter`: Footer with links
- `htmlAccordion`: Accordion/collapsible sections

#### Forms & Data
- `htmlForm`: Form with configurable fields
- `htmlTable`: Data table with headers and rows
- `htmlPagination`: Pagination controls

### Component Methods

```javascript
const components = new GUIXVisualComponents();

// Render component
const html = components.render('svgLineChart', {
  title: 'Sales',
  points: [[0, 10], [1, 20]]
});

// Get component metadata
const component = components.getComponent('cssButton');
// { name, format, category, accessibility, renderer }

// List components
const all = components.listComponents();      // All component names
const svg = components.getComponentsByFormat('svg');
const layouts = components.getComponentsByCategory('layout');
```

---

## GUIXRenderer API

Multi-format rendering engine with AST-based conversion.

### Methods

#### render(content, targetFormat, options)

Render content in specified format.

```javascript
const result = renderer.render(
  '<div>Content</div>',
  'html',
  {
    optimize: false,
    minify: false,
    wrapInDocument: true,
    title: 'Page Title',
    lang: 'en'
  }
);

// Returns:
// {
//   output: string,
//   format: string,
//   size: number,
//   optimized: boolean,
//   minified: boolean,
//   timestamp: number
// }
```

#### convert(content, fromFormat, toFormat, options)

Convert between formats using AST.

```javascript
const result = renderer.convert(
  '<svg></svg>',
  'svg',
  'html'
);

// Returns: { output, format, converted, conversionPath }
```

#### Minification

```javascript
// Minify individual formats
const minSvg = renderer.minifySVG(svgContent);
const minCss = renderer.minifyCSS(cssContent);
const minHtml = renderer.minifyHTML(htmlContent);

// Get minification report
const report = renderer.getMinificationReport(content, 'svg');
// { originalSize, minifiedSize, reduction, reductionPercent }
```

#### Optimization

```javascript
// Optimize for each format
const optimized = renderer.optimizeCSS(cssContent);
const optimized = renderer.optimizeSVG(svgContent);
const optimized = renderer.optimizeHTML(htmlContent);
```

#### Utilities

**getSupportedFormats()**: Get list of supported formats
**isValidFormat(format)**: Check if format is valid
**getFormatMetadata(format)**: Get format information
**clearCache()**: Clear render cache
**getCacheStats()**: Get cache statistics

---

## Theming System

### Built-in Themes

#### Light Theme
```javascript
{
  name: 'light',
  colors: {
    primary: '#3b82f6',      // Blue
    secondary: '#7c3aed',    // Purple
    background: '#ffffff',
    text: '#1f2937',
    border: '#e5e7eb',
    success: '#10b981',      // Green
    warning: '#f59e0b',      // Amber
    danger: '#ef4444'        // Red
  }
}
```

#### Dark Theme
```javascript
{
  name: 'dark',
  colors: {
    primary: '#60a5fa',      // Light Blue
    secondary: '#a78bfa',    // Light Purple
    background: '#1f2937',
    text: '#f3f4f6',
    border: '#374151',
    success: '#34d399',      // Light Green
    warning: '#fbbf24',      // Light Amber
    danger: '#f87171'        // Light Red
  }
}
```

### Custom Themes

```javascript
guix.createTheme('brand', {
  colors: {
    primary: '#your-color',
    secondary: '#your-color',
    // ... all required colors
  },
  typography: {
    fontFamily: 'Your Font',
    fontSize: '16px',
    fontWeight: '400',
    lineHeight: '1.6'
  }
});
```

---

## Component Library

### SVG Line Chart

```javascript
guix.renderComponent('svgLineChart', {
  title: 'Monthly Sales',
  xLabel: 'Month',
  yLabel: 'Revenue',
  points: [[0, 100], [1, 150], [2, 120], [3, 200]]
}, {
  theme: 'light'
});
```

### CSS Grid Layout

```javascript
guix.renderComponent('cssGrid', {
  columns: 3,
  gap: '1rem',
  items: 12
});
```

### HTML Form

```javascript
guix.renderComponent('htmlForm', {
  title: 'Contact Form',
  fields: [
    { name: 'name', label: 'Full Name', type: 'text' },
    { name: 'email', label: 'Email', type: 'email' },
    { name: 'message', label: 'Message', type: 'textarea' }
  ]
});
```

---

## Performance Optimization

### Caching Strategy

Components are cached by:
- Component name
- Component data (JSON serialized)
- Applied theme

Disable caching:
```javascript
const guix = new GUIXHybridModel({ cacheEnabled: false });
```

### Minification & Optimization

```javascript
// Minify output
const result = guix.render(content, {
  minify: true,
  optimize: true
});

// Check impact
const report = guix.renderer.getMinificationReport(content, 'css');
console.log(`Reduced by ${report.reductionPercent}`);
```

### Tensor Metrics

Monitor performance:
```javascript
const metrics = guix.getTensorMetrics();
console.log(`Cache hit rate: ${metrics.cache.hitRate}`);
console.log(`Average render time: ${metrics.renderTime.average}ms`);
```

---

## Error Handling

```javascript
const result = guix.renderComponent('invalid', {});

if (result.error) {
  console.error('Render failed:', result.error);
} else {
  console.log('Rendered:', result.output);
}
```

---

## Integration Examples

### With Linguistic Core

```javascript
import { LinguisticCore } from './src/linguistic-core.js';
import { GUIXHybridModel } from './src/guix-hybrid-model.js';

const linguistic = new LinguisticCore();
const guix = new GUIXHybridModel();

// Process text semantically, then visualize
const analysis = linguistic.processText(text);
const visualization = guix.renderComponent('svgNetwork', {
  nodes: analysis.entities,
  edges: analysis.relationships
});
```

### With Phase 6 Metrics

```javascript
// Use GUIX to visualize metrics dashboards
const result = guix.renderComponent('svgLineChart', {
  points: metricsData.timeSeriesPoints,
  title: 'Performance Metrics'
});
```

---

## Accessibility Features

All components include WCAG 2.1 Level AA compliance:

- Semantic HTML markup
- ARIA labels and roles
- Color contrast ratios ≥ 4.5:1
- Keyboard navigation support
- Focus indicators
- Responsive design

Verify accessibility:
```javascript
const a11y = guix.getComponentAccessibility('htmlForm');
console.log(a11y.features.semanticMarkup);      // true
console.log(a11y.features.ariaLabels);          // true
console.log(a11y.wcagLevel);                    // 'AA'
```

---

## Version & Support

- **Version**: 7.0.0
- **Phase**: Phase 7 - GUIX Hybrid Model
- **Status**: Production Ready
- **Dependencies**: None (zero external dependencies)
- **Node.js**: ≥ 18

For more information, see [GUIX_COMPONENT_LIBRARY.md](./GUIX_COMPONENT_LIBRARY.md).
