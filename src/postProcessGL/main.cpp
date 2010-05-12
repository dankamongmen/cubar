/*
* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* This software and the information contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a Non-Disclosure Agreement.  Any reproduction or
* disclosure to any third party without the express written consent of
* NVIDIA is prohibited.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

/* 
    This example demonstrates the use of CUDA/OpenGL interoperability
    to post-process an image of a 3D scene generated in OpenGL.

    The basic steps are:
    1 - render the scene to the framebuffer
    2 - copy the image to a PBO (pixel buffer object)
    3 - map this PBO so that its memory is accessible from CUDA
    4 - run CUDA to process the image, writing to memory mapped from a second PBO
    6 - copy from result PBO to a texture
    7 - display the texture

    Press space to toggle the CUDA processing on/off.
    Press 'a' to toggle animation.
    Press '+' and '-' to increment and decrement blur radius
*/

#pragma warning(disable:4996)

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, GL
#include <GL/glew.h>
#include <GL/glut.h>

#include "../../common/inc/cutil.h"

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int window_width = 512;
unsigned int window_height = 512;
unsigned int image_width = 512;
unsigned int image_height = 512;

// pbo variables
GLuint pbo_source;
GLuint pbo_dest;
unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;

// (offscreen) render target
GLuint fbo;
GLuint tex_fbo;
GLuint tex_screen;

float rotate[3];

bool enable_cuda = true;
bool animate = true;
int blur_radius = 8;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C" void runTest( int argc, char** argv);
extern "C" void process( int pbo_in, int pbo_out, int width, int height, int radius);
extern "C" void pboRegister( int pbo );
extern "C" void pboUnregister( int pbo );

// GL functionality
CUTBoolean initGL();
void createPBO( GLuint* pbo);
void deletePBO( GLuint* pbo);
void createFBO( GLuint* fbo, GLuint* tex);
void deleteFBO( GLuint* fbo, GLuint* tex);
void createTexture(GLuint* tex_name, unsigned int size_x, unsigned int size_y);
void deleteTexture( GLuint* tex);

// rendering callbacks
void display();
void idle();
void keyboard( unsigned char key, int x, int y);
void reshape(int w, int h);
void mainMenu(int i);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) {

    runTest( argc, argv);

    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) {

    // Create GL context
    glutInit( &argc, argv);
    glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( "CUDA OpenGL post-processing");

    // initialize GL
    if( CUTFalse == initGL()) {
        return;
    }

    // register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);
    glutReshapeFunc( reshape);
    glutIdleFunc( idle);

    // create menu
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Toggle CUDA processing [ ]", ' ');
    glutAddMenuEntry("Toggle animation [a]", 'a');
    glutAddMenuEntry("Increment blur radius [=]", '=');
    glutAddMenuEntry("Decrement blur radius [-]", '-');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    // create pbo
    createPBO( &pbo_source);
    createPBO( &pbo_dest);

    // create fbo
    createFBO( &fbo, &tex_fbo);
    // create texture for blitting onto the screen
    createTexture( &tex_screen, image_width, image_height);

    // start rendering mainloop
    glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
initGL() {

    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported(
        "GL_VERSION_2_0 " 
        "GL_ARB_pixel_buffer_object "
        "GL_EXT_framebuffer_object "
		)) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return CUTFalse;
    }

    // default initialization
    glClearColor( 0.5, 0.5, 0.5, 1.0);
    glDisable( GL_DEPTH_TEST);

    // viewport
    glViewport( 0, 0, window_width, window_height);

    // projection
    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_LIGHT0);
    float red[] = { 1.0, 0.1, 0.1, 1.0 };
    float white[] = { 1.0, 1.0, 1.0, 1.0 };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0);

    CUT_CHECK_ERROR_GL();

    return CUTTrue;
}

////////////////////////////////////////////////////////////////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
void
createPBO( GLuint* pbo) {

    // set up vertex data parameter
    num_texels = image_width * image_height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;
    void *data = malloc(size_tex_data);

    // create buffer object
    glGenBuffers( 1, pbo);
    glBindBuffer( GL_ARRAY_BUFFER, *pbo);

    // buffer data
    glBufferData( GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
    free(data);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // attach this Buffer Object to CUDA
    pboRegister(*pbo);

    CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete PBO
////////////////////////////////////////////////////////////////////////////////
void
deletePBO( GLuint* pbo) {

    glBindBuffer( GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers( 1, pbo);
    CUT_CHECK_ERROR_GL();

    *pbo = 0;
}

// render a simple 3D scene
void renderScene()
{
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode( GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, -3.0);
    glRotatef(rotate[0], 1.0, 0.0, 0.0);
    glRotatef(rotate[1], 0.0, 1.0, 0.0);
    glRotatef(rotate[2], 0.0, 0.0, 1.0);

    glViewport(0, 0, 512, 512);

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);

    glutSolidTeapot(1.0);

    CUT_CHECK_ERROR_GL();
}

// copy image and process using CUDA
void processImage()
{
    // tell cuda we are going to get into these buffers
    pboUnregister(pbo_source);

    // read data into pbo

    // activate destination buffer
    glBindBuffer( GL_PIXEL_PACK_BUFFER_ARB, pbo_source);

    // read
    glReadPixels( 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, NULL); 

    // Done : re-register for Cuda
    pboRegister(pbo_source);

    // run the Cuda kernel
    process( pbo_source, pbo_dest, image_width, image_height, blur_radius);

    // blit convolved texture onto the screen

    // detach from CUDA
    pboUnregister(pbo_dest);

    // download texture from PBO
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);
    glBindTexture( GL_TEXTURE_2D, tex_screen);
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    CUT_CHECK_ERROR_GL();
}

void displayImage()
{
    // render a screen sized quad
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode( GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode( GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, window_width, window_height);

    glBegin( GL_QUADS);

    glTexCoord2f( 0.0, 0.0);
    glVertex3f( -1.0, -1.0, 0.5);

    glTexCoord2f( 1.0, 0.0);
    glVertex3f(  1.0, -1.0, 0.5);

    glTexCoord2f( 1.0, 1.0);
    glVertex3f(  1.0,  1.0, 0.5);

    glTexCoord2f( 0.0, 1.0);
    glVertex3f( -1.0,  1.0, 0.5);

    glEnd();

    glMatrixMode( GL_PROJECTION);
    glPopMatrix();

    glDisable( GL_TEXTURE_2D);
    glBindBuffer( GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // re-attach to CUDA
    pboRegister(pbo_dest);

    CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void
display() {

    // render the scene to the fbo
    //glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, fbo);

    renderScene();

    if (enable_cuda) {
//        glReadBuffer( GL_COLOR_ATTACHMENT0_EXT);
        processImage();

        // switch back to screen
        //glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);

        displayImage();
    }

    glutSwapBuffers();
}

void idle()
{
    if (animate) {
        rotate[0] += 0.2;
        rotate[1] += 0.6;
        rotate[2] += 1.0;
    }
    glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void
keyboard( unsigned char key, int /*x*/, int /*y*/) {

    switch( key) {
    case( 27) :
        deletePBO( &pbo_source);
        deletePBO( &pbo_dest);
        deleteFBO( &fbo, &tex_fbo);
        deleteTexture( &tex_screen);
        exit( 0);
    case ' ':
        enable_cuda ^= 1;
        break;
    case 'a':
        animate ^= 1;
        break;
    case '=':
    case '+':
        if (blur_radius < 16) blur_radius++;
        printf("radius = %d\n", blur_radius);
        break;
    case '-':
        if (blur_radius > 1) blur_radius--;
        printf("radius = %d\n", blur_radius);
        break;
    }
}

void reshape(int w, int h)
{
    window_width = w;
    window_height = w;
}

void mainMenu(int i)
{
  keyboard((unsigned char) i, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Create offscreen render target
////////////////////////////////////////////////////////////////////////////////
void
createFBO( GLuint* fbo, GLuint* tex) {

    // create a new fbo
    glGenFramebuffersEXT( 1, fbo);
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, *fbo);
    CUT_CHECK_ERROR_GL();

    // check if the fbo is valid
    if( ! glIsFramebufferEXT( *fbo)) {
        fprintf( stderr, "Framebuffer object creation failed.\n");
        fflush( stderr);

        return;
    }

    // create attachment
    createTexture( tex, image_width, image_height);

    CUT_CHECK_ERROR_GL();

    // attach texture to fbo
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT,
        GL_COLOR_ATTACHMENT0_EXT,
        GL_TEXTURE_2D, 
        *tex, 
        0);

    CUT_CHECK_ERROR_GL();

    // deactivate offsreen render target
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);

    CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Cleanup offscreen render target
////////////////////////////////////////////////////////////////////////////////
void
deleteFBO( GLuint* fbo, GLuint* tex) {

    glDeleteFramebuffersEXT( 1, fbo);
    CUT_CHECK_ERROR_GL();

    deleteTexture( tex);

    *fbo = 0;  
}

////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void
deleteTexture( GLuint* tex) {

    glDeleteTextures( 1, tex);
    CUT_CHECK_ERROR_GL();

    *tex = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void
createTexture( GLuint* tex_name, unsigned int size_x, unsigned int size_y) {

    // create a tex as attachment
    glGenTextures( 1, tex_name);
    glBindTexture( GL_TEXTURE_2D, *tex_name);

    // set basic parameters
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // buffer data
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    CUT_CHECK_ERROR_GL();
}
