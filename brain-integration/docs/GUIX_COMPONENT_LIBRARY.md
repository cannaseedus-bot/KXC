# GUIX Component Library Reference

Complete documentation of all 30+ pre-built components with examples and accessibility information.

## Table of Contents

1. [SVG Visualization Components](#svg-visualization-components)
2. [CSS Layout Components](#css-layout-components)
3. [CSS Styled Components](#css-styled-components)
4. [HTML Semantic Components](#html-semantic-components)
5. [HTML Content Components](#html-content-components)
6. [Component Patterns](#component-patterns)
7. [Accessibility Checklist](#accessibility-checklist)

---

## SVG Visualization Components

SVG components are vector-based, infinitely scalable, and perfect for charts, diagrams, and visualizations.

### Line Chart (`svgLineChart`)

**Format**: SVG  
**Category**: Visualization  
**Accessibility**: Title, desc, color-contrast

**Purpose**: Display trends over time with connected data points.

**Properties**:
```javascript
{
  title: 'Sales Trend',           // Chart title
  xLabel: 'Month',                // X-axis label
  yLabel: 'Revenue ($)',          // Y-axis label
  points: [                       // Array of [x, y] coordinates
    [0, 100],
    [1, 150],
    [2, 120],
    [3, 200]
  ]
}
```

**Example**:
```javascript
const chart = guix.renderComponent('svgLineChart', {
  title: 'Quarterly Performance',
  points: [[0, 50], [1, 75], [2, 100], [3, 85]],
  xLabel: 'Quarter',
  yLabel: 'Growth %'
});
```

**ASCII Visualization**:
```
    200 |
        |         •
    150 |      •  
        |   •
    100 | •
        |___________
        Q1 Q2 Q3 Q4
```

### Bar Chart (`svgBarChart`)

**Format**: SVG  
**Category**: Visualization  
**Accessibility**: Title, color-contrast

**Purpose**: Compare values across categories.

**Properties**:
```javascript
{
  title: 'Sales by Region',
  bars: [
    { label: 'North', value: 150 },
    { label: 'South', value: 120 },
    { label: 'East', value: 180 },
    { label: 'West', value: 140 }
  ]
}
```

**Example**:
```javascript
const chart = guix.renderComponent('svgBarChart', {
  title: 'Monthly Revenue',
  bars: [
    { label: 'Jan', value: 45 },
    { label: 'Feb', value: 52 },
    { label: 'Mar', value: 48 }
  ]
});
```

**ASCII Visualization**:
```
200 |     █
150 |  █  █  █
100 | █   █  █
  0 |__________
   Jan Feb Mar
```

### Pie Chart (`svgPieChart`)

**Format**: SVG  
**Category**: Visualization  
**Accessibility**: Title, color-contrast

**Purpose**: Show proportion of a whole.

**Properties**:
```javascript
{
  title: 'Market Share',
  slices: [
    { label: 'Product A', value: 35 },
    { label: 'Product B', value: 25 },
    { label: 'Product C', value: 20 },
    { label: 'Product D', value: 20 }
  ]
}
```

**Example**:
```javascript
guix.renderComponent('svgPieChart', {
  title: 'Q1 Revenue Distribution',
  slices: [
    { label: 'Services', value: 45 },
    { label: 'Products', value: 35 },
    { label: 'Consulting', value: 20 }
  ]
});
```

### Scatter Plot (`svgScatterPlot`)

**Format**: SVG  
**Category**: Visualization  
**Accessibility**: Title, color-contrast

**Purpose**: Show relationship between two variables.

**Properties**:
```javascript
{
  title: 'Correlation Analysis',
  points: [[x1, y1], [x2, y2], ...]
}
```

### Heatmap (`svgHeatmap`)

**Format**: SVG  
**Category**: Visualization  
**Accessibility**: Title, color-contrast

**Purpose**: Visualize intensity/frequency data in 2D matrix.

**Properties**:
```javascript
{
  title: 'User Activity Matrix',
  matrix: [
    [1, 5, 3],
    [8, 2, 9],
    [4, 7, 6]
  ]
}
```

**ASCII Visualization**:
```
        Low    Medium  High
Time 1   ▮      ▮▮▮▮▮  ▮▮▮
Time 2   ▮▮▮▮▮▮▮ ▮    ▮▮▮▮▮▮▮
Time 3   ▮▮▮▮   ▮▮▮▮▮▮ ▮▮▮▮
```

### Network Diagram (`svgNetwork`)

**Format**: SVG  
**Category**: Visualization  
**Accessibility**: Title, color-contrast, desc

**Purpose**: Display interconnected nodes and relationships.

**Properties**:
```javascript
{
  title: 'System Architecture',
  nodes: [
    { id: 'a', label: 'Server', x: 100, y: 100 },
    { id: 'b', label: 'Database', x: 300, y: 100 },
    { id: 'c', label: 'Cache', x: 200, y: 250 }
  ],
  edges: [
    { from: 'a', to: 'b' },
    { from: 'a', to: 'c' },
    { from: 'b', to: 'c' }
  ]
}
```

### Hierarchy Diagram (`svgHierarchy`)

**Format**: SVG  
**Category**: Visualization  
**Accessibility**: Title, color-contrast

**Purpose**: Show hierarchical relationships (org charts, taxonomies).

**Properties**:
```javascript
{
  title: 'Organization Structure',
  root: {
    label: 'CEO',
    children: [
      { label: 'VP Engineering' },
      { label: 'VP Sales' },
      { label: 'VP Operations' }
    ]
  }
}
```

### Flowchart (`svgFlowchart`)

**Format**: SVG  
**Category**: Visualization  
**Accessibility**: Title, color-contrast, desc

**Purpose**: Visualize process flow with decision points.

**Properties**:
```javascript
{
  title: 'Approval Workflow',
  steps: [
    { label: 'Submit', type: 'start' },
    { label: 'Review', type: 'process' },
    { label: 'Approved?', type: 'decision' },
    { label: 'Complete', type: 'end' }
  ]
}
```

### Gauge (`svgGauge`)

**Format**: SVG  
**Category**: Visualization  
**Accessibility**: Title, color-contrast

**Purpose**: Display progress/performance on a scale.

**Properties**:
```javascript
{
  title: 'CPU Usage',
  value: 75,           // Current value
  minValue: 0,         // Minimum
  maxValue: 100        // Maximum
}
```

**ASCII Visualization**:
```
    ⤢
  ╱   ╲
 │  75% │
  ╲   ╱
    ⤡
```

### Radar Chart (`svgRadar`)

**Format**: SVG  
**Category**: Visualization  
**Accessibility**: Title, color-contrast

**Purpose**: Compare multiple dimensions simultaneously.

**Properties**:
```javascript
{
  title: 'Skill Assessment',
  categories: ['Speed', 'Quality', 'Reliability', 'Cost', 'Support'],
  values: [85, 90, 78, 65, 92]
}
```

---

## CSS Layout Components

CSS components provide responsive layouts using modern CSS features.

### Grid Layout (`cssGrid`)

**Format**: CSS  
**Category**: Layout  
**Accessibility**: Responsive, semantic

**Purpose**: Create flexible grid-based layouts with configurable columns.

**Properties**:
```javascript
{
  columns: 3,              // Number of columns
  gap: '1rem',             // Space between items
  items: 9                 // Number of grid items
}
```

**Features**:
- Responsive on mobile (1 column)
- Configurable gap between items
- Equal-width columns
- Auto-wrapping content

**Example**:
```javascript
guix.renderComponent('cssGrid', {
  columns: 3,
  gap: '2rem',
  items: 12
});
```

### Flexbox Layout (`cssFlexbox`)

**Format**: CSS  
**Category**: Layout  
**Accessibility**: Responsive, flexible

**Purpose**: Create flexible 1D layouts with flex direction and alignment control.

**Properties**:
```javascript
{
  direction: 'row',              // 'row' or 'column'
  justifyContent: 'space-between', // Content alignment
  alignItems: 'center',          // Item alignment
  items: 3                       // Number of items
}
```

**Example**:
```javascript
guix.renderComponent('cssFlexbox', {
  direction: 'row',
  justifyContent: 'space-around',
  alignItems: 'flex-start',
  items: 5
});
```

### Sidebar Layout (`cssSidebar`)

**Format**: CSS  
**Category**: Layout  
**Accessibility**: Responsive, semantic

**Purpose**: Create 2-column layout with sidebar navigation.

**Features**:
- Fixed-width sidebar (250px)
- Responsive (sidebar hides on mobile)
- Dark sidebar with light content area
- Perfect for admin dashboards

**Example**:
```javascript
guix.renderComponent('cssSidebar');
```

---

## CSS Styled Components

Pre-styled interactive components with theming support.

### Navbar (`cssNavbar`)

**Format**: CSS  
**Category**: Component  
**Accessibility**: Semantic nav, keyboard nav

**Purpose**: Top navigation bar with brand and menu items.

**Properties**:
```javascript
{
  brand: 'My App',
  items: ['Home', 'About', 'Services', 'Contact']
}
```

### Button (`cssButton`)

**Format**: CSS  
**Category**: Component  
**Accessibility**: Focus states, keyboard nav

**Purpose**: Styled button with variants and sizes.

**Properties**:
```javascript
{
  label: 'Click Me',
  type: 'primary',           // 'primary', 'secondary', 'danger'
  size: 'md'                 // 'sm', 'md', 'lg'
}
```

**Variants**: Primary (blue), Secondary (gray), Danger (red)

### Card (`cssCard`)

**Format**: CSS  
**Category**: Component  
**Accessibility**: Semantic structure

**Purpose**: Container with header, body, and footer.

**Properties**:
```javascript
{
  title: 'Card Title',
  content: 'Card body content',
  footer: 'Card footer'      // Optional
}
```

### Modal (`cssModal`)

**Format**: CSS  
**Category**: Component  
**Accessibility**: Focus trap, backdrop

**Purpose**: Dialog overlay with backdrop and animations.

**Features**:
- Smooth slide-in animation
- Overlay backdrop
- Close button
- Footer with action buttons

**Example**:
```javascript
guix.renderComponent('cssModal', {
  title: 'Confirmation',
  content: 'Are you sure?'
});
```

### Alert (`cssAlert`)

**Format**: CSS  
**Category**: Component  
**Accessibility**: ARIA roles

**Purpose**: Status messages with dismissible option.

**Properties**:
```javascript
{
  message: 'Operation successful!',
  type: 'success',           // 'info', 'success', 'warning', 'danger'
  dismissible: true
}
```

**Types**:
- Info: Blue, informational messages
- Success: Green, successful operations
- Warning: Amber, caution messages
- Danger: Red, error messages

### Tooltip (`cssTooltip`)

**Format**: CSS  
**Category**: Component  
**Accessibility**: aria-label, keyboard nav

**Purpose**: Show contextual help on hover.

**Properties**:
```javascript
{
  content: 'Click to learn more',
  position: 'top'            // 'top', 'bottom', 'left', 'right'
}
```

### Dropdown (`cssDropdown`)

**Format**: CSS  
**Category**: Component  
**Accessibility**: Keyboard nav, role=menu

**Purpose**: Dropdown menu with multiple options.

**Properties**:
```javascript
{
  label: 'Actions',
  items: [
    'Edit',
    'Delete',
    'Archive'
  ]
}
```

---

## HTML Semantic Components

Fully semantic HTML with maximum accessibility.

### Navigation (`htmlNavigation`)

**Format**: HTML  
**Category**: Semantic  
**Accessibility**: Semantic nav, aria-label

**Purpose**: Semantic navigation structure.

**Properties**:
```javascript
{
  items: [
    { label: 'Home', href: '/' },
    { label: 'About', href: '/about' },
    { label: 'Contact', href: '/contact' }
  ]
}
```

### Article (`htmlArticle`)

**Format**: HTML  
**Category**: Semantic  
**Accessibility**: Semantic markup, datetime

**Purpose**: Blog post or article with metadata.

**Properties**:
```javascript
{
  title: 'Article Title',
  author: 'John Doe',
  date: '2024-01-15',
  content: 'Article content...'
}
```

### Aside (`htmlAside`)

**Format**: HTML  
**Category**: Semantic  
**Accessibility**: Semantic complementary content

**Purpose**: Supplementary content sidebar.

**Properties**:
```javascript
{
  title: 'Related Links',
  content: 'Sidebar content...'
}
```

### Footer (`htmlFooter`)

**Format**: HTML  
**Category**: Semantic  
**Accessibility**: Semantic footer

**Purpose**: Page footer with links and copyright.

**Properties**:
```javascript
{
  company: 'Acme Corp',
  year: 2024,
  links: [
    { label: 'Privacy', href: '/privacy' },
    { label: 'Terms', href: '/terms' }
  ]
}
```

### Form (`htmlForm`)

**Format**: HTML  
**Category**: Content  
**Accessibility**: Labels, semantic inputs, ARIA

**Purpose**: Accessible form with labeled inputs.

**Properties**:
```javascript
{
  title: 'Contact Form',
  fields: [
    { name: 'email', label: 'Email', type: 'email' },
    { name: 'subject', label: 'Subject', type: 'text' },
    { name: 'message', label: 'Message', type: 'textarea' }
  ]
}
```

**Features**:
- Proper label associations
- Required field markers
- Semantic HTML form elements

### Table (`htmlTable`)

**Format**: HTML  
**Category**: Content  
**Accessibility**: Caption, headers, th/td roles

**Purpose**: Structured data in rows and columns.

**Properties**:
```javascript
{
  caption: 'Sales Data',
  headers: ['Product', 'Q1', 'Q2', 'Q3'],
  rows: [
    ['Product A', '100', '120', '150'],
    ['Product B', '80', '95', '110']
  ]
}
```

**Features**:
- Table caption
- Proper thead/tbody structure
- Header row with th elements

### Accordion (`htmlAccordion`)

**Format**: HTML  
**Category**: Content  
**Accessibility**: details/summary semantic elements

**Purpose**: Collapsible sections for content organization.

**Properties**:
```javascript
{
  items: [
    { title: 'Section 1', content: 'Content 1' },
    { title: 'Section 2', content: 'Content 2' }
  ]
}
```

### Breadcrumb (`htmlBreadcrumb`)

**Format**: HTML  
**Category**: Content  
**Accessibility**: aria-current, semantic

**Purpose**: Navigation breadcrumb trail.

**Properties**:
```javascript
{
  items: [
    { label: 'Home', href: '/' },
    { label: 'Products', href: '/products' },
    { label: 'Electronics', href: null }  // Current page
  ]
}
```

### Tabs (`htmlTabs`)

**Format**: HTML  
**Category**: Content  
**Accessibility**: role=tablist, aria-selected

**Purpose**: Tabbed interface for content organization.

**Properties**:
```javascript
{
  tabs: [
    { id: 'tab1', label: 'Overview', content: '...' },
    { id: 'tab2', label: 'Details', content: '...' }
  ]
}
```

### Pagination (`htmlPagination`)

**Format**: HTML  
**Category**: Content  
**Accessibility**: aria-current, navigation

**Purpose**: Navigate through pages of content.

**Properties**:
```javascript
{
  currentPage: 2,
  totalPages: 10
}
```

---

## Component Patterns

### Combining Components

```javascript
// Create a dashboard with multiple components
const dashboard = `
  ${guix.renderComponent('htmlNavigation', {...}).output}
  <div style="display: grid; grid-template-columns: 1fr 3fr; gap: 2rem;">
    ${guix.renderComponent('cssSidebar', {...}).output}
    <div>
      ${guix.renderComponent('svgLineChart', {...}).output}
      ${guix.renderComponent('cssCard', {...}).output}
    </div>
  </div>
`;
```

### Theming Components

```javascript
// Apply theme to component
guix.setTheme('dark');
const darkButton = guix.renderComponent('cssButton', {
  label: 'Dark Theme'
}, {
  theme: 'dark'
});

guix.setTheme('light');
const lightButton = guix.renderComponent('cssButton', {
  label: 'Light Theme'
}, {
  theme: 'light'
});
```

### Optimizing Output

```javascript
// Minify and optimize component
const optimized = guix.renderComponent('svgLineChart', 
  { points: [...] },
  {
    optimize: true,
    minify: true
  }
);

console.log('Optimization report:', guix.renderer.getMinificationReport(
  optimized.output, 
  'svg'
));
```

---

## Accessibility Checklist

All components meet WCAG 2.1 Level AA:

### HTML Components
- ✓ Semantic HTML markup (nav, article, form, table, etc.)
- ✓ Proper heading hierarchy
- ✓ Form labels associated with inputs
- ✓ Keyboard navigation support
- ✓ Focus indicators visible
- ✓ ARIA labels where needed
- ✓ Color not sole means of communication
- ✓ Color contrast ≥ 4.5:1 for text

### SVG Components
- ✓ Title elements for chart identification
- ✓ Desc elements for complex diagrams
- ✓ Color contrast ≥ 3:1 for graphics
- ✓ Legend for color-coded data
- ✓ Text alternatives for visual elements

### CSS Components
- ✓ Focus states on interactive elements
- ✓ Sufficient color contrast
- ✓ Responsive design (mobile-first)
- ✓ Touch-friendly sizing (≥44x44px)
- ✓ Keyboard accessible
- ✓ Works without color
- ✓ Works without animations

### All Components
- ✓ Proper language attributes
- ✓ Character encoding specified
- ✓ Responsive viewport settings
- ✓ Sufficient time limits
- ✓ No seizure/animation risks
- ✓ Clear link text
- ✓ Error prevention/recovery
- ✓ Descriptive page titles

---

## Component Count Summary

- **SVG Components**: 10 (visualization)
- **CSS Layouts**: 3 (grid, flexbox, sidebar)
- **CSS Components**: 7 (navbar, button, card, modal, alert, tooltip, dropdown)
- **HTML Semantic**: 4 (nav, article, aside, footer)
- **HTML Content**: 6 (form, table, accordion, breadcrumb, tabs, pagination)

**Total**: 30 Production-Ready Components

---

For API details, see [guix-model.md](./guix-model.md).
