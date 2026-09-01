/**
 * @file kuhul.css.hlsl
 * @brief KUHUL CSS Domain Shader — GPU-native SCX Atomic CSS parser and style resolver
 *
 * Sub-Category: 0x11
 * Entry:        CS_KuhulCSSDispatch
 * Routed from:  kuhul.hlsl CS_KuhulDispatch (Yax opcode, word1[31:16] == 0x0011)
 *
 * This shader parses KUHUL CSS glyph streams (including SCX `--⟁` token properties
 * and `[⟁attr]` attribute selectors) on the GPU and produces resolved style vectors
 * written to StyleOutput. CP-1 UI_FOLD then composites styles onto the DOM box tree
 * produced by kuhul.html.hlsl.
 *
 * CSS Glyph Alphabet — derived from SCX Atomic CSS (html-samples/deepseek_html_20260324_38cfbe.html)
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Layout group    [Pop → CM-1 gate]          0x00–0x0F
 *   DISPLAY       0x00  — display property token
 *   POSITION      0x01  — position (static/relative/absolute/fixed/sticky)
 *   FLEX_BOX      0x02  — flex container declaration
 *   GRID_BOX      0x03  — grid container declaration
 *   FLOAT         0x04  — float (left/right/none)
 *   CLEAR         0x05  — clear (left/right/both)
 *   VISIBILITY    0x06  — visibility
 *   ISOLATION     0x07  — isolation
 *
 * Flow group      [Wo → COMPUTE_FOLD]        0x10–0x1F
 *   FLEX_DIR      0x10  — flex-direction
 *   FLEX_WRAP     0x11  — flex-wrap
 *   JUSTIFY_C     0x12  — justify-content
 *   ALIGN_ITEMS   0x13  — align-items
 *   ALIGN_SELF    0x14  — align-self
 *   ALIGN_CONTENT 0x15  — align-content
 *   GAP           0x16  — gap / column-gap / row-gap
 *   GRID_COLS     0x17  — grid-template-columns
 *   GRID_ROWS     0x18  — grid-template-rows
 *   GRID_AREA     0x19  — grid-area / grid-column / grid-row
 *   ORDER         0x1A  — order (flex/grid child)
 *   FLEX_GROW     0x1B  — flex-grow
 *   FLEX_SHRINK   0x1C  — flex-shrink
 *   FLEX_BASIS    0x1D  — flex-basis
 *
 * Alignment group [Yax → UI_FOLD]            0x20–0x2F
 *   TOP           0x20  — top
 *   RIGHT         0x21  — right
 *   BOTTOM        0x22  — bottom
 *   LEFT          0x23  — left
 *   INSET         0x24  — inset shorthand
 *   Z_INDEX       0x25  — z-index
 *   VERTICAL_A    0x26  — vertical-align
 *   TEXT_ANCHOR   0x27  — text-anchor (SVG)
 *
 * Spacing group   [Sek → STORAGE_FOLD]       0x30–0x3F
 *   MARGIN        0x30  — margin (all sides)
 *   MARGIN_T      0x31  — margin-top
 *   MARGIN_R      0x32  — margin-right
 *   MARGIN_B      0x33  — margin-bottom
 *   MARGIN_L      0x34  — margin-left
 *   MARGIN_X      0x35  — margin-inline (left+right)
 *   MARGIN_Y      0x36  — margin-block (top+bottom)
 *   PADDING       0x37  — padding (all sides)
 *   PADDING_T     0x38  — padding-top
 *   PADDING_R     0x39  — padding-right
 *   PADDING_B     0x3A  — padding-bottom
 *   PADDING_L     0x3B  — padding-left
 *   PADDING_X     0x3C  — padding-inline
 *   PADDING_Y     0x3D  — padding-block
 *
 * Border group    [K'ayab' → STORAGE_FOLD]   0x40–0x4F
 *   BORDER        0x40  — border shorthand
 *   BORDER_W      0x41  — border-width
 *   BORDER_S      0x42  — border-style
 *   BORDER_C      0x43  — border-color
 *   BORDER_R      0x44  — border-radius
 *   BORDER_T      0x45  — border-top
 *   BORDER_R2     0x46  — border-right
 *   BORDER_B      0x47  — border-bottom
 *   BORDER_L      0x48  — border-left
 *   OUTLINE       0x49  — outline
 *   BOX_SHADOW    0x4A  — box-shadow
 *   RING          0x4B  — SCX ring utility (outline-offset + box-shadow ring)
 *
 * Typography group [Xul → COMPUTE MoE]       0x50–0x5F
 *   FONT_SIZE     0x50  — font-size
 *   FONT_WEIGHT   0x51  — font-weight
 *   FONT_FAMILY   0x52  — font-family
 *   FONT_STYLE    0x53  — font-style / font-variant
 *   LINE_HEIGHT   0x54  — line-height
 *   LETTER_SPACE  0x55  — letter-spacing
 *   TEXT_ALIGN    0x56  — text-align
 *   TEXT_COLOR    0x57  — color
 *   TEXT_DECO     0x58  — text-decoration
 *   TEXT_TRANSFORM 0x59 — text-transform
 *   WHITE_SPACE   0x5A  — white-space
 *   WORD_BREAK    0x5B  — word-break / overflow-wrap
 *   CONTENT       0x5C  — content (pseudo-element)
 *   TRUNCATE      0x5D  — SCX truncate utility (overflow:hidden + text-overflow:ellipsis)
 *
 * Sizing group    [Chum → COMPUTE_FOLD]      0x60–0x6F
 *   WIDTH         0x60  — width
 *   HEIGHT        0x61  — height
 *   MIN_W         0x62  — min-width
 *   MAX_W         0x63  — max-width
 *   MIN_H         0x64  — min-height
 *   MAX_H         0x65  — max-height
 *   ASPECT        0x66  — aspect-ratio
 *   OBJECT_FIT    0x67  — object-fit
 *   OBJECT_POS    0x68  — object-position
 *   BOX_SIZING    0x69  — box-sizing
 *
 * Effects/BG group [K'in → COMPUTE_FOLD]    0x70–0x7F
 *   BG_COLOR      0x70  — background-color
 *   BG_IMAGE      0x71  — background-image
 *   BG_SIZE       0x72  — background-size
 *   BG_POS        0x73  — background-position
 *   BG_REPEAT     0x74  — background-repeat
 *   BG_CLIP       0x75  — background-clip
 *   OPACITY       0x76  — opacity
 *   FILTER        0x77  — filter
 *   BACKDROP      0x78  — backdrop-filter
 *   MIX_BLEND     0x79  — mix-blend-mode
 *   GRADIENT_LIN  0x7A  — linear-gradient()
 *   GRADIENT_RAD  0x7B  — radial-gradient()
 *   GRADIENT_CONIC 0x7C — conic-gradient()
 *   GLASSMORPHISM 0x7D  — SCX glass utility (bg rgba + backdrop-filter:blur)
 *
 * Transform group [Ok → COMPUTE_FOLD]        0x80–0x8F
 *   TRANSFORM     0x80  — transform shorthand
 *   TRANSLATE     0x81  — translate / translateX / translateY
 *   ROTATE        0x82  — rotate
 *   SCALE         0x83  — scale / scaleX / scaleY
 *   SKEW          0x84  — skewX / skewY
 *   ORIGIN        0x85  — transform-origin
 *   TRANSITION    0x86  — transition
 *   ANIMATION     0x87  — animation
 *   KEYFRAME      0x88  — @keyframes block
 *   PERSPECTIVE   0x89  — perspective / perspective-origin
 *   WILL_CHANGE   0x8A  — will-change (GPU layer hint)
 *
 * Interactivity   [Chok → UI_FOLD]           0x90–0x9F
 *   CURSOR        0x90  — cursor
 *   POINTER_EVT   0x91  — pointer-events
 *   USER_SELECT   0x92  — user-select
 *   RESIZE        0x93  — resize
 *   APPEARANCE    0x94  — appearance
 *   TOUCH_ACTION  0x95  — touch-action
 *   SCROLL_SNAP   0x96  — scroll-snap-type / scroll-snap-align
 *   HOVER_STATE   0x97  — :hover pseudo-class binding
 *   FOCUS_STATE   0x98  — :focus / :focus-visible pseudo-class
 *   ACTIVE_STATE  0x99  — :active pseudo-class
 *
 * SCX ⟁ token group [special → STORAGE_FOLD] 0xA0–0xAF
 *   SCX_BG        0xA0  — --⟁bg / --⟁bg2 / --⟁bg3 reference
 *   SCX_FG        0xA1  — --⟁fg / --⟁fg2 reference
 *   SCX_AC        0xA2  — --⟁ac1 / --⟁ac2 / --⟁ac3 / --⟁ac4 reference
 *   SCX_SPACE     0xA3  — --⟁s[0-7] spacing token (pdata = scale index)
 *   SCX_RADIUS    0xA4  — --⟁r[1-5,full] radius token (pdata = scale index)
 *   SCX_FONT_SZ   0xA5  — --⟁fs[1-6] font-size token (pdata = scale index)
 *   SCX_SHADOW    0xA6  — --⟁shadow / --⟁shadow-lg / --⟁shadow-xl
 *   SCX_ATTR      0xA7  — [⟁attr] attribute selector (pdata = attr token ID)
 *   SCX_CUSTOM    0xAF  — arbitrary --⟁ custom property (pdata = property ID)
 *
 * Selector group  [implicit → CM-1 gate]     0xB0–0xBF
 *   SEL_CLASS     0xB0  — .class-name selector
 *   SEL_ID        0xB1  — #id selector
 *   SEL_ELEMENT   0xB2  — element selector
 *   SEL_PSEUDO    0xB3  — pseudo-class/element selector
 *   SEL_ATTR      0xB4  — [attr] attribute selector
 *   SEL_CHILD     0xB5  — > child combinator
 *   SEL_DESC      0xB6  — (space) descendant combinator
 *   SEL_SIBLING   0xB7  — ~ / + sibling combinators
 *   SEL_ROOT      0xB8  — :root (custom property scope)
 *   SEL_MEDIA     0xB9  — @media query block
 *   SEL_LAYER     0xBA  — @layer declaration
 *   SEL_CONTAINER 0xBB  — @container query
 *   SEL_DARK      0xBC  — [data-theme="dark"] / prefers-color-scheme:dark
 *   SEL_LIGHT     0xBD  — [data-theme="light"]
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Fold routing (category field from KPI1Instruction.word0[7:0]):
 *   0x00–0x07  → CM-1 CONTROL_FOLD  (layout/display/gate)
 *   0x10–0x1F  → COMPUTE_FOLD MM-1  (flow/flex/grid)
 *   0x20–0x2F  → UI_FOLD CP-1       (alignment/position)
 *   0x30–0x3F  → STORAGE_FOLD SM-1  (spacing/margin/padding)
 *   0x40–0x4F  → STORAGE_FOLD SM-1  (border)
 *   0x50–0x5F  → COMPUTE_FOLD MM-1  (typography via MoE)
 *   0x60–0x6F  → COMPUTE_FOLD MM-1  (sizing)
 *   0x70–0x7F  → COMPUTE_FOLD MM-1  (effects/background)
 *   0x80–0x8F  → COMPUTE_FOLD MM-1  (transform/animation)
 *   0x90–0x9F  → UI_FOLD CP-1       (interactivity)
 *   0xA0–0xAF  → STORAGE_FOLD SM-1  (SCX ⟁ token resolution)
 *   0xB0–0xBF  → CM-1 CONTROL_FOLD  (selectors/gate)
 */

// ============================================================================
// Shared structures (import from kuhul.hlsl)
// ============================================================================

struct KPI1Instruction
{
    uint word0;  // [7:0]=category_id [15:8]=glyph_id [23:16]=ptype [31:24]=plen
    uint word1;  // [7:0]=cdeg [15:8]=pos [31:16]=sub_category
};

// Resolved style record — written to StyleOutput per DOM node
struct StyleRecord
{
    uint  node_id;          // DOM node this style applies to
    uint  property_id;      // CSS property glyph (0x00–0xBF)
    float value_f;          // float value (font-size, opacity, width, etc.)
    uint  value_u;          // packed uint (color RGBA, flags, token refs)
    uint  specificity;      // selector specificity score
    uint  source_layer;     // @layer order index (0 = unscoped)
};

// SCX ⟁ token table entry (constant buffer, 64 entries max)
struct SCXToken
{
    float spacing[8];       // --⟁s0..s7  (rem values × 16)
    float radius[6];        // --⟁r1..r5 + r-full (9999 = pill)
    float font_size[6];     // --⟁fs1..fs6 (rem values × 16)
    uint  bg[3];            // --⟁bg / bg2 / bg3 packed RGBA
    uint  fg[2];            // --⟁fg / fg2
    uint  accent[4];        // --⟁ac1..ac4
    uint  shadow[3];        // --⟁shadow / shadow-lg / shadow-xl (packed flags)
};

// ============================================================================
// Constant Buffers
// ============================================================================

cbuffer CSSDispatchConstants : register(b0)
{
    uint  g_InstrCount;      // total KPI1 instructions in stream
    uint  g_NodeCount;       // DOM nodes to style
    uint  g_StyleOutputBase; // base index in StyleOutput UAV
    uint  g_SCXTokenOffset;  // index into SCXTokenBuffer for this frame
    float g_Viewport[2];     // viewport width / height (px)
    uint  g_ThemeFlags;      // bit0=dark, bit1=light, bit2=highcontrast
    uint  g_LayerCount;      // number of @layer declarations
};

// ============================================================================
// Buffers
// ============================================================================

StructuredBuffer<KPI1Instruction> InstrBuffer  : register(t0);  // KPI1 stream
StructuredBuffer<uint>            NodeBuffer   : register(t1);  // DOM node IDs
StructuredBuffer<SCXToken>        SCXTokenBuf  : register(t2);  // ⟁ token table
RWStructuredBuffer<StyleRecord>   StyleOutput  : register(u0);  // resolved styles
RWStructuredBuffer<uint>          DebugLog     : register(u1);  // debug/trace

// ============================================================================
// Helpers
// ============================================================================

uint PackRGBA(float r, float g, float b, float a)
{
    return (uint(a * 255) << 24) | (uint(b * 255) << 16)
         | (uint(g * 255) << 8) | uint(r * 255);
}

// Decode pdata field from instruction (bits [31:24] of word0)
uint GetPData(KPI1Instruction instr) { return (instr.word0 >> 24) & 0xFF; }
uint GetGlyph(KPI1Instruction instr) { return (instr.word0 >> 8)  & 0xFF; }
uint GetCat  (KPI1Instruction instr) { return  instr.word0        & 0xFF; }
uint GetPos  (KPI1Instruction instr) { return (instr.word1 >> 8)  & 0xFF; }

// Map pdata spacing index → SCX token value
float ResolveSCXSpacing(SCXToken tok, uint idx)
{
    if (idx >= 8) return 0.0f;
    return tok.spacing[idx] / 16.0f;  // stored as rem × 16
}

float ResolveSCXRadius(SCXToken tok, uint idx)
{
    if (idx >= 6) return 0.0f;
    float v = tok.radius[idx] / 16.0f;
    return (idx == 5) ? 9999.0f : v;  // r-full → pill
}

float ResolveSCXFontSize(SCXToken tok, uint idx)
{
    if (idx >= 6) return 1.0f;
    return tok.font_size[idx] / 16.0f;
}

// ============================================================================
// Per-glyph dispatch functions
// ============================================================================

StyleRecord DispatchLayout(KPI1Instruction instr, uint node_id, uint spec)
{
    StyleRecord rec;
    rec.node_id     = node_id;
    rec.property_id = GetGlyph(instr);
    rec.specificity = spec;
    rec.source_layer = 0;
    rec.value_f     = 0.0f;
    rec.value_u     = 0u;

    uint glyph = GetGlyph(instr);
    uint pdata = GetPData(instr);

    // DISPLAY: pdata encodes display mode
    // 0=block 1=flex 2=grid 3=inline 4=inline-flex 5=inline-grid 6=none 7=contents
    if (glyph == 0x00) { rec.value_u = pdata; }

    // POSITION: 0=static 1=relative 2=absolute 3=fixed 4=sticky
    else if (glyph == 0x01) { rec.value_u = pdata; }

    // FLEX_BOX: sets display:flex + direction from pdata (0=row 1=col 2=row-rev 3=col-rev)
    else if (glyph == 0x02) { rec.value_u = (1u) | (pdata << 8); }

    // GRID_BOX: sets display:grid
    else if (glyph == 0x03) { rec.value_u = 2u; }

    return rec;
}

StyleRecord DispatchFlow(KPI1Instruction instr, uint node_id, uint spec)
{
    StyleRecord rec;
    rec.node_id      = node_id;
    rec.property_id  = GetGlyph(instr);
    rec.specificity  = spec;
    rec.source_layer = 0;
    rec.value_f      = 0.0f;
    rec.value_u      = 0u;

    uint glyph = GetGlyph(instr);
    uint pdata = GetPData(instr);

    // FLEX_DIR: 0=row 1=col 2=row-reverse 3=col-reverse
    if      (glyph == 0x10) { rec.value_u = pdata; }
    // FLEX_WRAP: 0=nowrap 1=wrap 2=wrap-reverse
    else if (glyph == 0x11) { rec.value_u = pdata; }
    // JUSTIFY_C: 0=start 1=end 2=center 3=between 4=around 5=evenly
    else if (glyph == 0x12) { rec.value_u = pdata; }
    // ALIGN_ITEMS: 0=stretch 1=start 2=end 3=center 4=baseline
    else if (glyph == 0x13) { rec.value_u = pdata; }
    // ALIGN_SELF: same enum as align-items
    else if (glyph == 0x14) { rec.value_u = pdata; }
    // GAP: pdata = spacing scale index (maps to --⟁s[n])
    else if (glyph == 0x16) { rec.value_f = (float)pdata / 16.0f; }

    return rec;
}

StyleRecord DispatchSpacing(KPI1Instruction instr, uint node_id, uint spec,
                             SCXToken scxTok)
{
    StyleRecord rec;
    rec.node_id      = node_id;
    rec.property_id  = GetGlyph(instr);
    rec.specificity  = spec;
    rec.source_layer = 0;
    rec.value_f      = 0.0f;
    rec.value_u      = 0u;

    uint glyph = GetGlyph(instr);
    uint pdata = GetPData(instr);

    // All spacing glyphs: pdata[3:0] = scale index, pdata[7:4] = side flags
    float spacing_val = ResolveSCXSpacing(scxTok, pdata & 0x0F);
    rec.value_f = spacing_val;

    return rec;
}

StyleRecord DispatchBorder(KPI1Instruction instr, uint node_id, uint spec,
                            SCXToken scxTok)
{
    StyleRecord rec;
    rec.node_id      = node_id;
    rec.property_id  = GetGlyph(instr);
    rec.specificity  = spec;
    rec.source_layer = 0;
    rec.value_f      = 0.0f;
    rec.value_u      = 0u;

    uint glyph = GetGlyph(instr);
    uint pdata = GetPData(instr);

    // BORDER_R: pdata = radius scale index
    if (glyph == 0x44)
    {
        rec.value_f = ResolveSCXRadius(scxTok, pdata & 0x07);
    }
    // BOX_SHADOW: pdata = shadow tier (0=sm 1=base 2=lg 3=xl)
    else if (glyph == 0x4A)
    {
        rec.value_u = scxTok.shadow[min(pdata, 2u)];
    }

    return rec;
}

StyleRecord DispatchTypography(KPI1Instruction instr, uint node_id, uint spec,
                                SCXToken scxTok)
{
    StyleRecord rec;
    rec.node_id      = node_id;
    rec.property_id  = GetGlyph(instr);
    rec.specificity  = spec;
    rec.source_layer = 0;
    rec.value_f      = 0.0f;
    rec.value_u      = 0u;

    uint glyph = GetGlyph(instr);
    uint pdata = GetPData(instr);

    // FONT_SIZE: pdata = --⟁fs scale index (0-5)
    if (glyph == 0x50)
    {
        rec.value_f = ResolveSCXFontSize(scxTok, pdata & 0x07);
    }
    // FONT_WEIGHT: pdata encodes weight (1=100 ... 9=900)
    else if (glyph == 0x51)
    {
        rec.value_f = (float)(pdata * 100);
    }
    // TEXT_COLOR: value_u = RGBA (from accent table or custom)
    else if (glyph == 0x57)
    {
        // pdata[3:0]=source 0=fg 1=fg2 2=ac1 3=ac2 4=ac3 5=ac4
        if      (pdata == 0) rec.value_u = scxTok.fg[0];
        else if (pdata == 1) rec.value_u = scxTok.fg[1];
        else if (pdata >= 2 && pdata <= 5) rec.value_u = scxTok.accent[pdata - 2];
    }
    // LINE_HEIGHT: pdata / 10.0 (e.g. pdata=16 → 1.6)
    else if (glyph == 0x54)
    {
        rec.value_f = (float)pdata / 10.0f;
    }

    return rec;
}

StyleRecord DispatchEffects(KPI1Instruction instr, uint node_id, uint spec,
                             SCXToken scxTok)
{
    StyleRecord rec;
    rec.node_id      = node_id;
    rec.property_id  = GetGlyph(instr);
    rec.specificity  = spec;
    rec.source_layer = 0;
    rec.value_f      = 0.0f;
    rec.value_u      = 0u;

    uint glyph = GetGlyph(instr);
    uint pdata = GetPData(instr);

    // BG_COLOR: pdata selects from token table
    // 0=bg 1=bg2 2=bg3 3=ac1 4=ac2 5=ac3 6=ac4
    if (glyph == 0x70)
    {
        if      (pdata < 3)  rec.value_u = scxTok.bg[pdata];
        else if (pdata < 7)  rec.value_u = scxTok.accent[pdata - 3];
    }
    // OPACITY: pdata / 100.0
    else if (glyph == 0x76)
    {
        rec.value_f = (float)pdata / 100.0f;
    }
    // GLASSMORPHISM: SCX glass utility — value_u packs blur(px) + alpha
    else if (glyph == 0x7D)
    {
        uint blur_px = (pdata & 0x0F) * 4;   // 0-60px in steps of 4
        uint alpha   = (pdata >> 4) * 16;    // 0-240 alpha
        rec.value_u  = (blur_px << 8) | alpha;
    }

    return rec;
}

StyleRecord DispatchSCXToken(KPI1Instruction instr, uint node_id, uint spec,
                              SCXToken scxTok)
{
    StyleRecord rec;
    rec.node_id      = node_id;
    rec.property_id  = GetGlyph(instr);
    rec.specificity  = spec;
    rec.source_layer = 0;
    rec.value_f      = 0.0f;
    rec.value_u      = 0u;

    uint glyph = GetGlyph(instr);
    uint pdata = GetPData(instr);

    // SCX_SPACE: resolve --⟁s[pdata]
    if      (glyph == 0xA3) { rec.value_f = ResolveSCXSpacing(scxTok, pdata);  }
    // SCX_RADIUS: resolve --⟁r[pdata]
    else if (glyph == 0xA4) { rec.value_f = ResolveSCXRadius(scxTok, pdata);   }
    // SCX_FONT_SZ: resolve --⟁fs[pdata]
    else if (glyph == 0xA5) { rec.value_f = ResolveSCXFontSize(scxTok, pdata); }
    // SCX_BG: --⟁bg / bg2 / bg3
    else if (glyph == 0xA0) { rec.value_u = scxTok.bg[min(pdata, 2u)];         }
    // SCX_FG: --⟁fg / fg2
    else if (glyph == 0xA1) { rec.value_u = scxTok.fg[min(pdata, 1u)];         }
    // SCX_AC: --⟁ac1..ac4
    else if (glyph == 0xA2) { rec.value_u = scxTok.accent[min(pdata, 3u)];     }
    // SCX_SHADOW: --⟁shadow tiers
    else if (glyph == 0xA6) { rec.value_u = scxTok.shadow[min(pdata, 2u)];     }

    return rec;
}

// ============================================================================
// Main Entry Point
// ============================================================================

[numthreads(64, 1, 1)]
void CS_KuhulCSSDispatch(uint3 DTid : SV_DispatchThreadID)
{
    uint instr_idx = DTid.x;
    if (instr_idx >= g_InstrCount)
        return;

    KPI1Instruction instr = InstrBuffer[instr_idx];
    uint sub_cat = (instr.word1 >> 16) & 0xFFFF;

    // Verify this instruction belongs to the CSS domain (sub-category 0x11)
    if (sub_cat != 0x0011)
        return;

    uint cat     = GetCat(instr);
    uint node_id = GetPos(instr);  // DOM node index encoded in pos field
    uint spec    = instr.word1 & 0xFF;  // specificity from cdeg field

    SCXToken scxTok = SCXTokenBuf[g_SCXTokenOffset];

    StyleRecord rec;
    rec.node_id      = node_id;
    rec.property_id  = GetGlyph(instr);
    rec.specificity  = spec;
    rec.source_layer = 0;
    rec.value_f      = 0.0f;
    rec.value_u      = 0u;

    // ── Route to fold handler by category range ───────────────────────────
    if      (cat <= 0x07) rec = DispatchLayout   (instr, node_id, spec);
    else if (cat <= 0x1F) rec = DispatchFlow     (instr, node_id, spec);
    else if (cat <= 0x2F)
    {
        // Yax → UI_FOLD alignment: emit as-is with pdata as float
        rec.value_f = (float)GetPData(instr);
    }
    else if (cat <= 0x3F) rec = DispatchSpacing  (instr, node_id, spec, scxTok);
    else if (cat <= 0x4F) rec = DispatchBorder   (instr, node_id, spec, scxTok);
    else if (cat <= 0x5F) rec = DispatchTypography(instr, node_id, spec, scxTok);
    else if (cat <= 0x6F)
    {
        // Chum sizing: value_f encodes viewport-relative or absolute width/height
        rec.value_f = (float)GetPData(instr) / 100.0f * g_Viewport[cat == 0x61 ? 1 : 0];
    }
    else if (cat <= 0x7F) rec = DispatchEffects   (instr, node_id, spec, scxTok);
    else if (cat <= 0x8F)
    {
        // Ok transform: emit glyph + pdata; transform matrix computed by CP-1
        rec.value_u = GetPData(instr);
    }
    else if (cat <= 0x9F)
    {
        // Chok interactivity: pdata = enum value, stored as uint
        rec.value_u = GetPData(instr);
    }
    else if (cat <= 0xAF) rec = DispatchSCXToken(instr, node_id, spec, scxTok);
    else if (cat <= 0xBF)
    {
        // Selector: stored as-is; specificity accumulation handled by CM-1 gate
        rec.value_u = GetPData(instr);
    }

    // Write to StyleOutput UAV
    uint out_idx = g_StyleOutputBase + instr_idx;
    StyleOutput[out_idx] = rec;

    // Debug trace: record glyph + cat + node
    DebugLog[instr_idx] = (cat << 24) | (GetGlyph(instr) << 16) | (node_id & 0xFFFF);
}
