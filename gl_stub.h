#ifndef GL_STUB_H
#define GL_STUB_H

// GL/glut stub definitions for compiling without the actual libraries

// GLUT constants
#define GLUT_RGBA 0
#define GLUT_DOUBLE 2
#define GLUT_DEPTH 16
#define GLUT_ALPHA 8

// GLUT callback function types
typedef void (*GLUTDisplayFunc)(void);
typedef void (*GLUTReshapeFunc)(int width, int height);
typedef void (*GLUTKeyboardFunc)(unsigned char key, int x, int y);
typedef void (*GLUTMouseFunc)(int button, int state, int x, int y);
typedef void (*GLUTMotionFunc)(int x, int y);
typedef void (*GLUTIdleFunc)(void);

// Stub GLUT functions
void glutInit(int* argc, char** argv);
void glutInitDisplayMode(unsigned int mode);
void glutInitWindowSize(int width, int height);
void glutInitWindowPosition(int x, int y);
int glutCreateWindow(const char* title);
void glutDisplayFunc(GLUTDisplayFunc func);
void glutReshapeFunc(GLUTReshapeFunc func);
void glutKeyboardFunc(GLUTKeyboardFunc func);
void glutMouseFunc(GLUTMouseFunc func);
void glutMotionFunc(GLUTMotionFunc func);
void glutIdleFunc(GLUTIdleFunc func);
void glutMainLoop(void);
void glutSwapBuffers(void);
void glutPostRedisplay(void);

// OpenGL data types
typedef float GLfloat;
typedef unsigned int GLuint;
typedef int GLint;
typedef unsigned int GLenum;
typedef unsigned char GLubyte;

// OpenGL constants
#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_DEPTH_BUFFER_BIT 0x00000100
#define GL_TRIANGLES 0x0004
#define GL_QUADS 0x0007
#define GL_LINES 0x0001
#define GL_POINTS 0x0000
#define GL_MODELVIEW 0x1700
#define GL_PROJECTION 0x1701
#define GL_LIGHTING 0x0B50
#define GL_LIGHT0 0x4000
#define GL_LIGHT1 0x4001
#define GL_AMBIENT 0x1200
#define GL_DIFFUSE 0x1201
#define GL_SPECULAR 0x1202
#define GL_POSITION 0x1203
#define GL_SMOOTH 0x1D01
#define GL_COMPILE 0x1300

// OpenGL functions
void glClear(GLuint mask);
void glClearColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
void glMatrixMode(GLenum mode);
void glLoadIdentity(void);
void glBegin(GLenum mode);
void glEnd(void);
void glVertex3f(GLfloat x, GLfloat y, GLfloat z);
void glVertex2f(GLfloat x, GLfloat y);
void glColor3f(GLfloat red, GLfloat green, GLfloat blue);
void glColor4f(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
void glTranslatef(GLfloat x, GLfloat y, GLfloat z);
void glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z);
void glScalef(GLfloat x, GLfloat y, GLfloat z);
void glPushMatrix(void);
void glPopMatrix(void);
void glEnable(GLenum cap);
void glDisable(GLenum cap);
GLuint glGenLists(GLint range);
void glNewList(GLuint list, GLenum mode);
void glEndList(void);
void glCallList(GLuint list);

#endif // GL_STUB_H
