// =========================
// renderer_gl.hpp
// =========================
#include "kernels.hpp"
#include "simulation.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

// ---------------- GLUT ----------------
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

// ---------------- Globals ----------------
static char titleBuf[256];
static GLuint gPBO = 0;
static GLuint gTex = 0;
static cudaGraphicsResource* gCudaPBO = nullptr;
static Simulation* gSim = nullptr;
struct uchar4;

// ---------------- Set window title ----------------
static inline void setWindowTitleFromParams(const Params& p, const char* mode) {
    std::snprintf(titleBuf, sizeof(titleBuf), "%s - t=%.1f (%d epochs): energy=%.3f, skyrmion=%.3f, vortex=%.3f", mode, p.epochs * p.time_step, p.epochs, p.energy, p.skyrmion, p.vortex);
    glutSetWindowTitle(titleBuf);
}

// ---------------- Display mode for window ----------------
static void screen_display(uchar4* d_out, double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p) {
    switch (p.display_mode) {
        case 1:
            visualization::show_density(d_out, d_en, d_entmp, d_Field, d1fd1x, d_gridsum, h_gridsum, p);
            setWindowTitleFromParams(p, "Energy");
            break;
        case 2:
            visualization::show_magnetization(d_out, d_Field, p);
            setWindowTitleFromParams(p, "Magnetization");
            break;
        case 3:
            visualization::show_vortex1(d_out, d_en, d_entmp, d_Field, d_gridsum, h_gridsum, p);
            setWindowTitleFromParams(p, "OP1");
            break;
        case 4:
            visualization::show_vortex2(d_out, d_en, d_entmp, d_Field, d_gridsum, h_gridsum, p);
            setWindowTitleFromParams(p, "OP2");
            break;
        case 5:
            visualization::show_magnetic_flux(d_out, d_en, d_entmp, d_Field, d1fd1x, d_gridsum, h_gridsum, p);
            setWindowTitleFromParams(p, "Flux");
            break;
        default:
            visualization::show_density(d_out, d_en, d_entmp, d_Field, d1fd1x, d_gridsum, h_gridsum, p);
            setWindowTitleFromParams(p, "Energy");
            break;
    }
}

// ---------------- Draw a GL textrue quad ----------------
static void drawTextureQuad(const Params& p) {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, gTex);
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(0,        0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(0,        p.ylen);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(p.xlen,   p.ylen);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(p.xlen,   0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

// ---------------- GLUT callbacks ----------------
static void display() {
    if (!gSim) return;
    auto& S = *gSim;

    // Map CUDA PBO
    uchar4* d_out = nullptr;
    size_t bytes  = 0;
    cudaGraphicsMapResources(1, &gCudaPBO, 0);
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_out), &bytes, gCudaPBO);

    // First render (before stepping)
    screen_display(d_out, S.d.en, S.d.entmp, S.d.Field, S.d.d1fd1x, S.d.gridsum, S.h.gridsum, S.p);

    // Optional stepping
    if (S.p.newtonflow) {
        for (int i = 0; i < S.p.iters_per_render; ++i) {
            S.p.error = S.step();
            S.computeObservables();
            S.p.epochs++;
        }
        screen_display(d_out, S.d.en, S.d.entmp, S.d.Field, S.d.d1fd1x, S.d.gridsum, S.h.gridsum, S.p);
        std::printf("t=%.1f (%d epochs): error=%.6f, energy=%.6f, skyrmion=%.6f, vortex=%.6f\n", S.p.epochs * S.p.time_step, S.p.epochs, S.p.error, S.p.energy, S.p.skyrmion, S.p.vortex);
        // Output data
        if(S.p.output_results){
            S.output_data();
            S.p.output_results = false;
        }
    }

    cudaGraphicsUnmapResources(1, &gCudaPBO, 0);

    // Upload from PBO into texture (no reallocation)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gPBO);
    glBindTexture(GL_TEXTURE_2D, gTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, S.p.xlen, S.p.ylen, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Draw textured quad in pixel coords
    drawTextureQuad(S.p);
    glutSwapBuffers();
    glutPostRedisplay();

    if (S.p.error <= S.p.tolerance || S.p.epochs >= S.p.loops_max) {
        glutLeaveMainLoop();   // this closes the GLUT loop cleanly
        return;
    }
}

// ---------------- Reshape the screen window ----------------
static void reshape(int w, int h) {
    if (!gSim) return;
    const int W = gSim->p.xlen, H = gSim->p.ylen;

    // Lock window to simulation size for 1:1 pixels
    if (w != W || h != H) {
        glutReshapeWindow(W, H);
        return;
    }
    glViewport(0, 0, W, H);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, W, H, 0, -1, 1); // coordinates in texture pixels
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

// ---------------- Keyboard allocations ----------------
static void keyboard(unsigned char key, int, int) {
    if (!gSim) return;
    auto& p = gSim->p;
    switch (key) {
        case 'n': p.newtonflow = !p.newtonflow; break;
        case 'k': p.killkinen  = !p.killkinen;  break;
        case 'o': p.output_results = true;      break;
        case 27:  std::exit(0);
    }
}

// ---------------- Special keyboard allocations (display mode selector) ----------------
static void specialKeys(int key, int, int) {
    if (!gSim) return;
    auto& p = gSim->p;
    if (key == GLUT_KEY_F1) p.display_mode = 1;
    if (key == GLUT_KEY_F2) p.display_mode = 2;
    if (key == GLUT_KEY_F3) p.display_mode = 3;
    if (key == GLUT_KEY_F4) p.display_mode = 4;
    if (key == GLUT_KEY_F5) p.display_mode = 5;
}

// ---------------- Mouse function allocation ----------------
static void mouse(int button, int state, int x, int y) {
    if (!gSim) return;
    auto& S = *gSim;

    // With window locked to xlen×ylen, x/y already map 1:1 to pixels
    int px = x;
    int py = y;
    if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) {
        if (S.p.soliton_id == 0)
            initial_configuration::create_skyrmion(py, px, S.p.skyrmion_rotation, S.d.Field, S.d.grid, S.p);
        if (S.p.soliton_id == 1)
            initial_configuration::create_vortex(py, px, S.p.vortex_type, S.d.Field, S.d.grid, S.p);
        if (S.p.soliton_id == 2) {
            initial_configuration::create_skyrmion(py, px, S.p.skyrmion_rotation, S.d.Field, S.d.grid, S.p);
            initial_configuration::create_vortex(py, px, S.p.vortex_type, S.d.Field, S.d.grid, S.p);
        }
        S.computeObservables();
    }
}

// ---------------- Exit clear up ----------------
static void onExit() {
    if (gCudaPBO) { cudaGraphicsUnregisterResource(gCudaPBO); gCudaPBO = nullptr; }
    if (gPBO) { glDeleteBuffers(1, &gPBO); gPBO = 0; }
    if (gTex) { glDeleteTextures(1, &gTex); gTex = 0; }
    if (gSim)  { gSim->uninit(); }
}

// ---------------- Right-click menu callbacks ----------------
static void mymenu(int value) {
    if (!gSim) return;
    auto& p = gSim->p;
    switch (value) {
        case 0: return;
        case 1: p.soliton_id = 0; break; // Skyrmion
        case 2: p.soliton_id = 1; break; // Vortex
        case 3: p.soliton_id = 2; break; // SVP
    }
    glutMouseFunc(mouse);
}

// ---------------- Skyrmion menu option ----------------
static void skyrmionOption(int value) {
    if (!gSim) return;
    auto& p = gSim->p;
    p.soliton_id = 0;
    switch (value) {
        case 0: p.skyrmion_rotation = 0.0;       break;
        case 1: p.skyrmion_rotation = M_PI / 2.; break;
        case 2: p.skyrmion_rotation = M_PI;      break;
    }
    glutMouseFunc(mouse);
}

// ---------------- Vortex menu options ----------------
static void vortexOption(int value) {
    if (!gSim) return;
    auto& p = gSim->p;
    p.soliton_id = 1;
    switch (value) {
        case 0: p.vortex_type = 0; break; // Vortex
        case 1: p.vortex_type = 1; break; // Antivortex
    }
    glutMouseFunc(mouse);
}

// ---------------- Composite SVP menu option ----------------
static void svpOption(int value) {
    if (!gSim) return;
    auto& p = gSim->p;
    p.soliton_id = 2;
    switch (value) {
        case 0: p.skyrmion_rotation = 0.0;       break;
        case 1: p.skyrmion_rotation = M_PI / 2.; break;
        case 2: p.skyrmion_rotation = M_PI;      break;
    }
    glutMouseFunc(mouse);
}

// ---------------- Right-click menu creator ----------------
static void createMenu() {
    int vortexMenu = glutCreateMenu(vortexOption);
    glutAddMenuEntry("Vortex",     0);
    glutAddMenuEntry("Antivortex", 1);

    int skyrmionMenu = glutCreateMenu(skyrmionOption);
    glutAddMenuEntry("Skyrmion, angle=0",    0);
    glutAddMenuEntry("Skyrmion, angle=pi/2", 1);
    glutAddMenuEntry("Skyrmion, angle=pi",   2);

    int svpMenu = glutCreateMenu(svpOption);
    glutAddMenuEntry("Skyrmion-vortex pair, angle=0",    0);
    glutAddMenuEntry("Skyrmion-vortex pair, angle=pi/2", 1);
    glutAddMenuEntry("Skyrmion-vortex pair, angle=pi",   2);

    glutCreateMenu(mymenu); // Object selection
    glutAddSubMenu("Skyrmion",             skyrmionMenu);
    glutAddSubMenu("Vortex",               vortexMenu);
    glutAddSubMenu("Skyrmion-vortex pair", svpMenu);
    glutAttachMenu(GLUT_RIGHT_BUTTON); // right-click for menu
}


// ---------------- Init GL and entry ----------------
static void initGL(Simulation& S, int* argc, char** argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(S.p.xlen, S.p.ylen);       // 1:1 window size
    glutCreateWindow("--== Ferromagnetic Superconductors ==--");
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION); // returns to main loop after closing

    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::fprintf(stderr, "GLEW init failed: %s\n", glewGetErrorString(err));
        std::exit(1);
    }

    // Create texture once (no per-frame reallocation)
    glGenTextures(1, &gTex);
    glBindTexture(GL_TEXTURE_2D, gTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, S.p.xlen, S.p.ylen, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // PBO sized to exactly 4 * x * y
    glGenBuffers(1, &gPBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gPBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_t(S.p.xlen) * size_t(S.p.ylen) * 4u, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register for CUDA interop
    cudaGraphicsGLRegisterBuffer(&gCudaPBO, gPBO, cudaGraphicsMapFlagsWriteDiscard);

    // Set 1:1 projection
    reshape(S.p.xlen, S.p.ylen);

    // GLUT callbacks
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeys);
	createMenu();           // ← build right-click menu
	glutMouseFunc(mouse);   // ensure mouse callback is set

    atexit(onExit);
}

// ---------------- Instructions ----------------
void printInstructions() {
	printf("Controls:\n"
		"Energy density display:						F1\n"
		"Magnetization display:						F2\n"
		"Order parameter 1 display:					F3\n"
        "Order parameter 1 display:					F4\n"
        "Magnetic flux display:						F5\n"
		"Toggle arrested Newton flow:					n\n"
		"Toggle arresting criteria:					k\n"
		"Soliton selection menu:						Hold right-click\n"
		"Place soliton:							Right-click\n\n");
}

// ---------------- Simulation with GL ----------------
extern "C" void run_gl_simulation(Simulation& sim, int* argc, char** argv) {
    gSim = &sim;
    initGL(sim, argc, argv);
    printInstructions();
    glutMainLoop();
}
