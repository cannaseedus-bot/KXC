/**
 * @file kuhul.html.hlsl
 * @brief KUHUL HTML Domain Shader — GPU-native HTML glyph parser and DOM layout engine
 *
 * Sub-Category: 0x10
 * Entry:        CS_KuhulHTMLDispatch
 * Routed from:  kuhul.hlsl CS_KuhulDispatch (Yax opcode, word1[31:16] == 0x0010)
 *
 * This shader parses KUHUL HTML glyph streams on the GPU and produces a flat
 * DOM box tree written to RenderOutput. CP-1 UI_FOLD then composites the box
 * tree into the final token/text/pixel output stream.
 *
 * HTML Glyph Alphabet:
 * ─────────────────────────────────────────────────────────────────────────────
 * Structural tokens:
 *   OPEN        0xF0  — open tag begin (followed by element glyph)
 *   CLOSE       0xF1  — close tag (matched by element glyph)
 *   SELF_CLOSE  0xF2  — self-closing tag (e.g. <img/>)
 *   ATTR        0xF3  — attribute key=value pair follows
 *   TEXT        0xF4  — text content follows (uint token IDs)
 *   DOCTYPE     0xF5  — <!DOCTYPE html>
 *   COMMENT     0xF6  — <!-- comment -->
 *
 * Block elements:
 *   DIV         0x10  — generic block container
 *   P           0x11  — paragraph
 *   H1          0x12  — heading 1
 *   H2          0x13  — heading 2
 *   H3          0x14  — heading 3
 *   H4          0x15  — heading 4
 *   H5          0x16  — heading 5
 *   H6          0x17  — heading 6
 *   SECTION     0x18  — semantic section
 *   ARTICLE     0x19  — semantic article
 *   HEADER      0x1A  — page/section header
 *   FOOTER      0x1B  — page/section footer
 *   MAIN        0x1C  — main content
 *   NAV         0x1D  — navigation
 *   ASIDE       0x1E  — sidebar
 *   UL          0x1F  — unordered list
 *   OL          0x20  — ordered list
 *   LI          0x21  — list item
 *   TABLE       0x22  — table
 *   TR          0x23  — table row
 *   TD          0x24  — table cell
 *   TH          0x25  — table header cell
 *   PRE         0x26  — preformatted text
 *   BLOCKQUOTE  0x27  — blockquote
 *   HR          0x28  — horizontal rule (self-closing)
 *   FIGURE      0x29  — figure container
 *   FIGCAPTION  0x2A  — figure caption
 *
 * Inline elements:
 *   SPAN        0x30  — generic inline
 *   A           0x31  — hyperlink
 *   STRONG      0x32  — bold
 *   EM          0x33  — italic
 *   CODE        0x34  — inline code
 *   BR          0x35  — line break (self-closing)
 *   IMG         0x36  — image (self-closing)
 *   LABEL       0x37  — form label
 *   ABBR        0x38  — abbreviation
 *   TIME        0x39  — datetime
 *   MARK        0x3A  — highlighted text
 *   SUB         0x3B  — subscript
 *   SUP         0x3C  — superscript
 *
 * Form elements:
 *   FORM        0x40  — form container
 *   INPUT       0x41  — input field (self-closing)
 *   TEXTAREA    0x42  — multiline text
 *   BUTTON      0x43  — button
 *   SELECT      0x44  — dropdown
 *   OPTION      0x45  — dropdown option
 *   FIELDSET    0x46  — form group
 *   LEGEND      0x47  — fieldset legend
 *
 * Media elements:
 *   VIDEO       0x50  — video
 *   AUDIO       0x51  — audio
 *   SOURCE      0x52  — media source (self-closing)
 *   CANVAS      0x53  — canvas (links to COMPUTE_FOLD for GPU draw)
 *   SVG         0x54  — inline SVG (routes to kuhul.svg.hlsl sub-dispatch)
 *   IFRAME      0x55  — embedded frame
 *
 * Head elements:
 *   HTML        0x60  — root html element
 *   HEAD        0x61  — document head
 *   BODY        0x62  — document body
 *   TITLE       0x63  — document title
 *   META_TAG    0x64  — <meta> tag (self-closing)
 *   LINK_TAG    0x65  — <link> tag (self-closing)
 *   SCRIPT      0x66  — <script> (routes to kuhul.wasm.hlsl or Muwan)
 *   STYLE       0x67  — <style> (routes to kuhul.css.hlsl)
 *
 * Attribute glyph IDs (follow ATTR structural token):
 *   ATTR_ID         0xA0
 *   ATTR_CLASS      0xA1
 *   ATTR_HREF       0xA2
 *   ATTR_SRC        0xA3
 *   ATTR_ALT        0xA4
 *   ATTR_TYPE       0xA5
 *   ATTR_NAME       0xA6
 *   ATTR_VALUE      0xA7
 *   ATTR_PLACEHOLDER 0xA8
 *   ATTR_STYLE      0xA9  — inline style (routes to kuhul.css.hlsl)
 *   ATTR_DATA       0xAA  — data-* attribute
 *   ATTR_ARIA       0xAB  — aria-* attribute
 *   ATTR_EVENT      0xAC  — on* event handler → Muwan FOREIGN_CALL
 *
 * DOM Box Tree Output Format (RenderOutput):
 *   Each DOM node writes 8 words to RenderOutput:
 *   [0] node_id       — unique within this frame
 *   [1] parent_id     — parent node_id (0 = root)
 *   [2] element_glyph — element type (DIV=0x10, P=0x11, etc.)
 *   [3] display_type  — 0=block 1=inline 2=none 3=flex 4=grid
 *   [4] text_token_id — first text token (if TEXT node), else 0
 *   [5] text_length   — number of text tokens
 *   [6] attr_flags    — bitmask of present attributes (id/class/href/src/style...)
 *   [7] child_count   — number of direct children
 */

// ============================================================================
// Resource Bindings
// ============================================================================

StructuredBuffer<uint>  HTMLGlyphs     : register(t0);  // HTML glyph stream (from kuhul.hlsl KPI1 payload)
StructuredBuffer<uint>  HTMLPayloads   : register(t1);  // attribute values, text tokens, href strings
StructuredBuffer<uint>  ControlFlags   : register(t2);  // CM-1 gate
StructuredBuffer<uint>  CSSStyleMap    : register(t3);  // computed styles from kuhul.css.hlsl (if pre-run)

RWStructuredBuffer<uint> RenderOutput  : register(u0);  // DOM box tree (8 words per node)
RWStructuredBuffer<uint> TextTokenBuf  : register(u1);  // text content token IDs → CP-1 UI_FOLD
RWStructuredBuffer<uint> EventBuf      : register(u2);  // event handler registrations → Muwan IPC
RWStructuredBuffer<uint> SubDispatch   : register(u3);  // SVG/CSS/WASM sub-dispatch signals

cbuffer HTMLParams : register(b0)
{
    uint num_glyphs;       // total glyph tokens in HTMLGlyphs
    uint payload_base;     // base offset in HTMLPayloads
    uint max_nodes;        // max DOM nodes this frame (RenderOutput capacity / 8)
    uint cm1_gate;         // must == 0x0001
    uint frame_id;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// ============================================================================
// Element Glyph Constants
// ============================================================================

// Structural
static const uint G_OPEN       = 0xF0u;
static const uint G_CLOSE      = 0xF1u;
static const uint G_SELF_CLOSE = 0xF2u;
static const uint G_ATTR       = 0xF3u;
static const uint G_TEXT       = 0xF4u;
static const uint G_DOCTYPE    = 0xF5u;
static const uint G_COMMENT    = 0xF6u;

// Block
static const uint G_DIV        = 0x10u;
static const uint G_P          = 0x11u;
static const uint G_H1         = 0x12u;
static const uint G_H2         = 0x13u;
static const uint G_H3         = 0x14u;
static const uint G_H4         = 0x15u;
static const uint G_H5         = 0x16u;
static const uint G_H6         = 0x17u;
static const uint G_SECTION    = 0x18u;
static const uint G_ARTICLE    = 0x19u;
static const uint G_HEADER     = 0x1Au;
static const uint G_FOOTER     = 0x1Bu;
static const uint G_MAIN       = 0x1Cu;
static const uint G_NAV        = 0x1Du;
static const uint G_ASIDE      = 0x1Eu;
static const uint G_UL         = 0x1Fu;
static const uint G_OL         = 0x20u;
static const uint G_LI         = 0x21u;
static const uint G_TABLE      = 0x22u;
static const uint G_TR         = 0x23u;
static const uint G_TD         = 0x24u;
static const uint G_TH         = 0x25u;
static const uint G_PRE        = 0x26u;
static const uint G_BLOCKQUOTE = 0x27u;
static const uint G_HR         = 0x28u;
static const uint G_FIGURE     = 0x29u;
static const uint G_FIGCAPTION = 0x2Au;

// Inline
static const uint G_SPAN       = 0x30u;
static const uint G_A          = 0x31u;
static const uint G_STRONG     = 0x32u;
static const uint G_EM         = 0x33u;
static const uint G_CODE       = 0x34u;
static const uint G_BR         = 0x35u;
static const uint G_IMG        = 0x36u;
static const uint G_LABEL      = 0x37u;
static const uint G_MARK       = 0x3Au;

// Form
static const uint G_FORM       = 0x40u;
static const uint G_INPUT      = 0x41u;
static const uint G_TEXTAREA   = 0x42u;
static const uint G_BUTTON     = 0x43u;
static const uint G_SELECT     = 0x44u;
static const uint G_OPTION     = 0x45u;

// Media / Special
static const uint G_VIDEO      = 0x50u;
static const uint G_CANVAS     = 0x53u;
static const uint G_SVG        = 0x54u;

// Head
static const uint G_HTML       = 0x60u;
static const uint G_HEAD       = 0x61u;
static const uint G_BODY       = 0x62u;
static const uint G_TITLE      = 0x63u;
static const uint G_SCRIPT     = 0x66u;
static const uint G_STYLE      = 0x67u;

// Attribute IDs
static const uint A_ID         = 0xA0u;
static const uint A_CLASS      = 0xA1u;
static const uint A_HREF       = 0xA2u;
static const uint A_SRC        = 0xA3u;
static const uint A_TYPE       = 0xA5u;
static const uint A_STYLE      = 0xA9u;
static const uint A_EVENT      = 0xACu;

// Display type codes (stored in DOM box word[3])
static const uint DISP_BLOCK   = 0u;
static const uint DISP_INLINE  = 1u;
static const uint DISP_NONE    = 2u;
static const uint DISP_FLEX    = 3u;
static const uint DISP_GRID    = 4u;

// ============================================================================
// Display type lookup (element glyph → default CSS display)
// ============================================================================

uint DefaultDisplayType(uint glyph)
{
    // Block elements
    if (glyph >= G_DIV && glyph <= G_FIGCAPTION) return DISP_BLOCK;
    if (glyph == G_FORM) return DISP_BLOCK;
    if (glyph == G_HTML || glyph == G_HEAD || glyph == G_BODY) return DISP_BLOCK;
    // Inline elements
    if (glyph >= G_SPAN && glyph <= G_MARK) return DISP_INLINE;
    if (glyph == G_LABEL) return DISP_INLINE;
    // Form inline
    if (glyph == G_INPUT || glyph == G_BUTTON || glyph == G_SELECT) return DISP_INLINE;
    // Hidden by default
    if (glyph == G_HEAD || glyph == G_SCRIPT || glyph == G_STYLE) return DISP_NONE;
    // Media block
    if (glyph >= G_VIDEO && glyph <= G_SVG) return DISP_BLOCK;
    return DISP_BLOCK;
}

// ============================================================================
// Groupshared DOM stack (parse stack for nested elements)
// Max depth = 64 (matches numthreads)
// ============================================================================

groupshared uint gs_stack[64];        // parent node_id stack
groupshared uint gs_stack_depth;      // current nesting depth
groupshared uint gs_node_counter;     // monotonic node_id within frame
groupshared uint gs_text_counter;     // text token write pointer
groupshared uint gs_event_counter;    // event registration counter
groupshared uint gs_gate_ok;          // CM-1 gate

// ============================================================================
// Main HTML Dispatch Kernel
// ============================================================================
// Each thread group processes a CHUNK of the glyph stream sequentially.
// Thread 0 manages the parse stack; other threads handle parallel attribute
// and text payload extraction within each element scope.

[numthreads(64, 1, 1)]
void CS_KuhulHTMLDispatch(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    // CM-1 gate
    if (tid == 0)
        gs_gate_ok = (ControlFlags[0] == cm1_gate) ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    if (gs_gate_ok == 0u) return;

    // Initialize stack on first group
    if (gid.x == 0 && tid == 0)
    {
        gs_stack_depth  = 0u;
        gs_node_counter = 0u;
        gs_text_counter = 0u;
        gs_event_counter= 0u;
        gs_stack[0]     = 0u;  // root parent = 0
    }
    GroupMemoryBarrierWithGroupSync();

    // Each group handles a window of glyphs: [gid.x*64 .. gid.x*64+63]
    uint glyph_start = gid.x * 64u;
    uint glyph_end   = min(glyph_start + 64u, num_glyphs);

    // Thread 0 walks the parse window sequentially (DOM tree is sequential by nature)
    if (tid == 0)
    {
        for (uint gi = glyph_start; gi < glyph_end; gi++)
        {
            uint g = HTMLGlyphs[gi];

            // ── DOCTYPE / COMMENT: skip ─────────────────────────────────────
            if (g == G_DOCTYPE || g == G_COMMENT)
                continue;

            // ── OPEN tag: push a new DOM node ───────────────────────────────
            if (g == G_OPEN && gi + 1 < num_glyphs)
            {
                uint elem_glyph = HTMLGlyphs[gi + 1];
                gi++;  // consume element glyph

                uint node_id  = ++gs_node_counter;
                uint parent_id = (gs_stack_depth > 0) ? gs_stack[gs_stack_depth - 1] : 0u;

                // Write DOM box (8 words)
                if (node_id < max_nodes)
                {
                    uint base = node_id * 8u;
                    RenderOutput[base + 0] = node_id;
                    RenderOutput[base + 1] = parent_id;
                    RenderOutput[base + 2] = elem_glyph;
                    RenderOutput[base + 3] = DefaultDisplayType(elem_glyph);
                    RenderOutput[base + 4] = 0u;  // text_token_id (filled on TEXT)
                    RenderOutput[base + 5] = 0u;  // text_length
                    RenderOutput[base + 6] = 0u;  // attr_flags
                    RenderOutput[base + 7] = 0u;  // child_count

                    // Increment parent's child_count
                    if (parent_id > 0 && parent_id < max_nodes)
                        RenderOutput[parent_id * 8u + 7]++;
                }

                // Push node onto parse stack (not self-closing)
                bool self_closing = (elem_glyph == G_HR  || elem_glyph == G_BR  ||
                                     elem_glyph == G_IMG || elem_glyph == G_INPUT ||
                                     elem_glyph == G_SELF_CLOSE);
                if (!self_closing && gs_stack_depth < 63u)
                    gs_stack[gs_stack_depth++] = node_id;

                // Sub-dispatch: CANVAS → COMPUTE_FOLD, SVG → kuhul.svg.hlsl, STYLE → kuhul.css.hlsl
                if (elem_glyph == G_CANVAS)
                    SubDispatch[node_id % 64u] = 0x0002u;  // FDISPATCH_COMPUTE
                else if (elem_glyph == G_SVG)
                    SubDispatch[node_id % 64u] = 0x0012u;  // sub-cat 0x12 = kuhul.svg.hlsl
                else if (elem_glyph == G_STYLE)
                    SubDispatch[node_id % 64u] = 0x0011u;  // sub-cat 0x11 = kuhul.css.hlsl
                else if (elem_glyph == G_SCRIPT)
                    SubDispatch[node_id % 64u] = 0x0016u;  // sub-cat 0x16 = kuhul.wasm.hlsl
            }

            // ── SELF_CLOSE: node already pushed as non-stack element ────────
            else if (g == G_SELF_CLOSE)
            {
                // no stack push needed — already handled in OPEN path
            }

            // ── CLOSE tag: pop parse stack ──────────────────────────────────
            else if (g == G_CLOSE)
            {
                if (gs_stack_depth > 0)
                    gs_stack_depth--;
                gi++;  // consume the closing element glyph (must match, not validated here)
            }

            // ── ATTR: set attribute flag on current open node ───────────────
            else if (g == G_ATTR && gi + 2 < num_glyphs)
            {
                uint attr_id  = HTMLGlyphs[gi + 1];
                uint attr_val = HTMLPayloads[payload_base + gi + 2];
                gi += 2;

                uint cur_node = (gs_stack_depth > 0) ? gs_stack[gs_stack_depth - 1] : 0u;
                if (cur_node > 0 && cur_node < max_nodes)
                {
                    uint attr_bit = 1u << (attr_id & 0x1Fu);
                    RenderOutput[cur_node * 8u + 6] |= attr_bit;

                    // Event attribute → register in EventBuf for Muwan IPC
                    if (attr_id == A_EVENT)
                    {
                        uint ei = gs_event_counter++;
                        EventBuf[ei * 4 + 0] = cur_node;
                        EventBuf[ei * 4 + 1] = attr_val;   // event type token
                        EventBuf[ei * 4 + 2] = frame_id;
                        EventBuf[ei * 4 + 3] = 0u;         // handler slot (filled by Muwan)
                    }
                }
            }

            // ── TEXT: write text tokens for current node ────────────────────
            else if (g == G_TEXT && gi + 1 < num_glyphs)
            {
                uint text_len = HTMLGlyphs[gi + 1];
                gi++;

                uint cur_node = (gs_stack_depth > 0) ? gs_stack[gs_stack_depth - 1] : 0u;
                uint text_start = gs_text_counter;

                // Copy text tokens to TextTokenBuf
                for (uint t = 0; t < text_len && (gi + 1 + t) < num_glyphs; t++)
                    TextTokenBuf[text_start + t] = HTMLPayloads[payload_base + gi + 1 + t];
                gs_text_counter += text_len;
                gi += text_len;

                // Record in DOM box
                if (cur_node > 0 && cur_node < max_nodes)
                {
                    RenderOutput[cur_node * 8u + 4] = text_start;   // text_token_id
                    RenderOutput[cur_node * 8u + 5] = text_len;      // text_length
                }

                // Text tokens → CP-1 UI_FOLD will render them as response output
                // TextTokenBuf is bound as t14 (DynamicShard14) in CS_UIFold
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Threads 1-63: parallel attribute value extraction for this glyph window
    // (Read HTMLPayloads in parallel to build style/href/src values)
    // Simplified: thread handles payload_base + gid.x*64 + tid slot
    uint payload_idx = payload_base + glyph_start + tid;
    if (tid > 0 && (glyph_start + tid) < num_glyphs)
    {
        // Payload word is already available in HTMLPayloads[payload_idx]
        // Full impl: map back to node and write computed attribute value
        // For now: pass-through (thread 0 already handled attribute flags)
        (void)HTMLPayloads[payload_idx];
    }
}
