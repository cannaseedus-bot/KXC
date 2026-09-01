// gl_harness.cpp — self-contained OpenGL 4.3 compute harness for Intel HD 4600.
// No GL headers/libs needed: dynamically loads opengl32.dll and wglGetProcAddress.
// Proves GL 4.3 compute works (SSBO in/out + dispatch + readback) in a clean path.
// Build (MSVC): cl /O2 gl_harness.cpp /Fe:gl_harness.exe
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <vector>

// ── GL base types (no gl.h available) ──────────────────────────────────────
typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef int GLsizei;
typedef int GLint;
typedef long long GLintptr;
typedef long long GLsizeiptr;
typedef unsigned int GLbitfield;
#ifndef APIENTRY
#define APIENTRY __stdcall
#endif

// ── GL constants (4.3 subset we need) ──────────────────────────────────────
#define GL_COMPUTE_SHADER 0x91B9
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_VERTEX_SHADER 0x8B31
#define GL_ARRAY_BUFFER 0x8892
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#define GL_STATIC_DRAW 0x88E4
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_ALL_BARRIER_BITS 0xFFFFFFFF
#define GL_MAJOR_VERSION 0x821B
#define GL_MINOR_VERSION 0x821C

// ── WGL (loaded from opengl32.dll) ─────────────────────────────────────────
typedef HGLRC (WINAPI *wglCreateContext_t)(HDC);
typedef BOOL  (WINAPI *wglMakeCurrent_t)(HDC, HGLRC);
typedef BOOL  (WINAPI *wglDeleteContext_t)(HGLRC);
typedef PROC  (WINAPI *wglGetProcAddress_t)(LPCSTR);
typedef HGLRC (WINAPI *wglCreateContextAttribsARB_t)(HDC, HGLRC, const int*);
typedef BOOL  (WINAPI *wglChoosePixelFormatARB_t)(HDC, const int*, const FLOAT*, UINT, int*, UINT*);

// ── GL functions (via wglGetProcAddress) ───────────────────────────────────
typedef GLuint (APIENTRY *glCreateShader_t)(GLuint);
typedef void   (APIENTRY *glShaderSource_t)(GLuint, GLsizei, const char**, const GLint*);
typedef void   (APIENTRY *glCompileShader_t)(GLuint);
typedef GLuint (APIENTRY *glCreateProgram_t)(void);
typedef void   (APIENTRY *glAttachShader_t)(GLuint, GLuint);
typedef void   (APIENTRY *glLinkProgram_t)(GLuint);
typedef void   (APIENTRY *glUseProgram_t)(GLuint);
typedef void   (APIENTRY *glGetShaderiv_t)(GLuint, GLenum, GLint*);
typedef void   (APIENTRY *glGetProgramiv_t)(GLuint, GLenum, GLint*);
typedef void   (APIENTRY *glGetShaderInfoLog_t)(GLuint, GLsizei, GLsizei*, char*);
typedef void   (APIENTRY *glGetProgramInfoLog_t)(GLuint, GLsizei, GLsizei*, char*);
typedef void   (APIENTRY *glGenBuffers_t)(GLsizei, GLuint*);
typedef void   (APIENTRY *glBindBuffer_t)(GLenum, GLuint);
typedef void   (APIENTRY *glBufferData_t)(GLenum, GLsizeiptr, const void*, GLenum);
typedef void   (APIENTRY *glBufferSubData_t)(GLenum, GLintptr, GLsizeiptr, const void*);
typedef void   (APIENTRY *glGetBufferSubData_t)(GLenum, GLintptr, GLsizeiptr, void*);
typedef void   (APIENTRY *glDispatchCompute_t)(GLuint, GLuint, GLuint);
typedef void   (APIENTRY *glMemoryBarrier_t)(GLbitfield);
typedef void   (APIENTRY *glGetIntegerv_t)(GLenum, GLint*);

static wglCreateContext_t wglCreateContext_fn;
static wglMakeCurrent_t wglMakeCurrent_fn;
static wglDeleteContext_t wglDeleteContext_fn;
static wglGetProcAddress_t wglGetProcAddress_fn;
static wglCreateContextAttribsARB_t wglCreateContextAttribsARB_fn;
static wglChoosePixelFormatARB_t wglChoosePixelFormatARB_fn;
static glCreateShader_t glCreateShader_fn; static glShaderSource_t glShaderSource_fn;
static glCompileShader_t glCompileShader_fn; static glCreateProgram_t glCreateProgram_fn;
static glAttachShader_t glAttachShader_fn; static glLinkProgram_t glLinkProgram_fn;
static glUseProgram_t glUseProgram_fn; static glGetShaderiv_t glGetShaderiv_fn;
static glGetProgramiv_t glGetProgramiv_fn; static glGetShaderInfoLog_t glGetShaderInfoLog_fn;
static glGetProgramInfoLog_t glGetProgramInfoLog_fn; static glGenBuffers_t glGenBuffers_fn;
static glBindBuffer_t glBindBuffer_fn; static glBufferData_t glBufferData_fn;
static glBufferSubData_t glBufferSubData_fn; static glGetBufferSubData_t glGetBufferSubData_fn;
static glDispatchCompute_t glDispatchCompute_fn; static glMemoryBarrier_t glMemoryBarrier_fn;
static glGetIntegerv_t glGetIntegerv_fn;

#define GLF(name) *(void**)&glCreateShader_fn = (void*)wglGetProcAddress_fn(#name)
static bool load_gl() {
    GLF(glCreateShader); GLF(glShaderSource); GLF(glCompileShader); GLF(glCreateProgram);
    GLF(glAttachShader); GLF(glLinkProgram); GLF(glUseProgram); GLF(glGetShaderiv);
    GLF(glGetProgramiv); GLF(glGetShaderInfoLog); GLF(glGetProgramInfoLog); GLF(glGenBuffers);
    GLF(glBindBuffer); GLF(glBufferData); GLF(glBufferSubData); GLF(glGetBufferSubData);
    GLF(glDispatchCompute); GLF(glMemoryBarrier); GLF(glGetIntegerv);
    if(!glCreateShader_fn) printf("  null: glCreateShader\n");
    if(!glDispatchCompute_fn) printf("  null: glDispatchCompute\n");
    if(!glGenBuffers_fn) printf("  null: glGenBuffers\n");
    if(!glBufferData_fn) printf("  null: glBufferData\n");
    if(!glGetBufferSubData_fn) printf("  null: glGetBufferSubData\n");
    return glCreateShader_fn && glDispatchCompute_fn && glGenBuffers_fn && glBufferData_fn && glGetBufferSubData_fn;
}

static GLuint compile_compute(const char* src) {
    GLuint s = glCreateShader_fn(GL_COMPUTE_SHADER);
    const char* p = src;
    glShaderSource_fn(s, 1, &p, nullptr);
    glCompileShader_fn(s);
    GLint ok=0; glGetShaderiv_fn(s, GL_COMPILE_STATUS, &ok);
    if(!ok){ char log[1024]; GLsizei len=0; glGetShaderInfoLog_fn(s,1024,&len,log);
        printf("SHADER COMPILE ERROR:\n%s\n", log); return 0; }
    GLuint prog = glCreateProgram_fn();
    glAttachShader_fn(prog, s);
    glLinkProgram_fn(prog);
    GLint lk=0; glGetProgramiv_fn(prog, GL_LINK_STATUS, &lk);
    if(!lk){ char log[1024]; GLsizei len=0; glGetProgramInfoLog_fn(prog,1024,&len,log);
        printf("PROGRAM LINK ERROR:\n%s\n", log); return 0; }
    return prog;
}

int main() {
    HMODULE gl = LoadLibraryA("opengl32.dll");
    if(!gl){ printf("FAIL: cannot load opengl32.dll\n"); return 1; }
    wglCreateContext_fn = (wglCreateContext_t)GetProcAddress(gl,"wglCreateContext");
    wglMakeCurrent_fn = (wglMakeCurrent_t)GetProcAddress(gl,"wglMakeCurrent");
    wglDeleteContext_fn = (wglDeleteContext_t)GetProcAddress(gl,"wglDeleteContext");
    wglGetProcAddress_fn = (wglGetProcAddress_t)GetProcAddress(gl,"wglGetProcAddress");

    // hidden window + DC
    WNDCLASSA wc{}; wc.lpfnWndProc = DefWindowProcA; wc.lpszClassName = "GLH";
    RegisterClassA(&wc);
    HWND hwnd = CreateWindowA("GLH","gl",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);
    HDC dc = GetDC(hwnd);
    PIXELFORMATDESCRIPTOR pfd{};
    pfd.nSize=sizeof(pfd); pfd.nVersion=1; pfd.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER;
    pfd.iPixelType=PFD_TYPE_RGBA; pfd.cColorBits=24; pfd.cDepthBits=24;
    int pf = ChoosePixelFormat(dc,&pfd);
    if(!pf){ printf("FAIL: ChoosePixelFormat\n"); return 1; }
    SetPixelFormat(dc,pf,&pfd);

    // context (try 4.3 core via ARB, fallback to default)
    HGLRC ctx = wglCreateContext_fn(dc);
    wglMakeCurrent_fn(dc,ctx);
    wglCreateContextAttribsARB_fn = (wglCreateContextAttribsARB_t)wglGetProcAddress_fn("wglCreateContextAttribsARB");
    if(wglCreateContextAttribsARB_fn){
        int attrs[] = {0x2091,4,0x2092,3,0x9126,1,0}; // MAJOR=4 MINOR=3 PROFILE=CORE
        HGLRC ctx43 = wglCreateContextAttribsARB_fn(dc,0,attrs);
        if(ctx43){ wglMakeCurrent_fn(dc,ctx43); wglDeleteContext_fn(ctx); ctx=ctx43; }
        else printf("  ARB 4.3 core context creation failed; using fallback context\n");
    } else printf("  wglCreateContextAttribsARB unavailable; using fallback context\n");
    // try again for a 4.3 context, and print which ARB versions succeed
    if(wglCreateContextAttribsARB_fn){
        for(int v=43;v>=40;v--){
            int attrs[] = {0x2091,v/10,0x2092,v%10,0}; // MAJOR MINOR (any profile)
            HGLRC t = wglCreateContextAttribsARB_fn(dc,0,attrs);
            if(t){ printf("  created context %d.%d\n", v/10, v%10);
                BOOL cur = wglMakeCurrent_fn(dc,t); wglDeleteContext_fn(ctx); ctx=t;
                printf("  wglMakeCurrent returned %d, error=%lu\n", cur, (unsigned long)GetLastError());
                break; }
        }
    }
    // base GL 1.x funcs are exported by opengl32.dll directly
    typedef const char* (APIENTRY *glGetString_t)(GLenum);
    auto glGetString_fn = (glGetString_t)GetProcAddress(gl,"glGetString");
    if(glGetString_fn){ const char* v=glGetString_fn(0x1F02); const char* r=glGetString_fn(0x1F03);
        printf("  GL_VERSION=%s GL_RENDERER=%s\n", v?v:"?", r?r:"?"); }
    else printf("  glGetString not exported by opengl32\n");
    if(!load_gl()){ printf("FAIL: cannot load GL 4.3 functions\n"); return 1; }

    GLint maj=0,min=0; glGetIntegerv_fn(GL_MAJOR_VERSION,&maj); glGetIntegerv_fn(GL_MINOR_VERSION,&min);
    printf("OpenGL context: %d.%d\n", maj, min);

    // compute: o = a + b, N=1024
    const char* src = "#version 430\n"
        "layout(local_size_x=64) in;\n"
        "layout(std430,binding=0) buffer A{float a[];};\n"
        "layout(std430,binding=1) buffer B{float b[];};\n"
        "layout(std430,binding=2) buffer O{float o[];};\n"
        "void main(){ uint i=gl_GlobalInvocationID.x; o[i]=a[i]+b[i]; }\n";
    GLuint prog = compile_compute(src);
    if(!prog){ printf("FAIL: shader\n"); return 1; }
    glUseProgram_fn(prog);

    const int N=1024;
    std::vector<float> a(N),b(N),o(N,0);
    for(int i=0;i<N;i++){ a[i]=float(i)*0.5f; b[i]=float(i)*0.25f; }
    GLuint bufs[3]; glGenBuffers_fn(3,bufs);
    glBindBuffer_fn(GL_SHADER_STORAGE_BUFFER,bufs[0]);
    glBufferData_fn(GL_SHADER_STORAGE_BUFFER,N*4,a.data(),GL_STATIC_DRAW);
    glBindBuffer_fn(GL_SHADER_STORAGE_BUFFER,bufs[1]);
    glBufferData_fn(GL_SHADER_STORAGE_BUFFER,N*4,b.data(),GL_STATIC_DRAW);
    glBindBuffer_fn(GL_SHADER_STORAGE_BUFFER,bufs[2]);
    glBufferData_fn(GL_SHADER_STORAGE_BUFFER,N*4,nullptr,GL_STATIC_DRAW);

    glDispatchCompute_fn((N+63)/64,1,1);
    glMemoryBarrier_fn(GL_ALL_BARRIER_BITS);
    glBindBuffer_fn(GL_SHADER_STORAGE_BUFFER,bufs[2]);
    glGetBufferSubData_fn(GL_SHADER_STORAGE_BUFFER,0,N*4,o.data());

    int ok=0; for(int i=0;i<N;i++) if(fabsf(o[i]-(a[i]+b[i]))<1e-3) ok++;
    printf("compute o=a+b: %d/%d match  o[0]=%f o[1023]=%f\n", ok,N,o[0],o[N-1]);
    printf("RESULT: %s\n", ok==N?"PASS (OpenGL 4.3 compute works on HD 4600)":"FAIL");

    wglMakeCurrent_fn(dc,nullptr); wglDeleteContext_fn(ctx); ReleaseDC(hwnd,dc);
    return ok==N?0:1;
}
