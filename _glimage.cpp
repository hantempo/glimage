#include <Python.h>
#include <numpy/arrayobject.h>
#include "glimage.h"

static char module_docstring[] =
    "This module provides some methods to convert pixels of images between different OpenGL formats.";

#define DEFINE_CONVERSION_WRAPPER(func_name, input_numpy_type, input_c_type, output_numpy_type, output_c_type) \
static PyObject *glimage_##func_name(PyObject *self, PyObject *args) \
{ \
    unsigned int width, height; \
    PyObject *input_pixels_obj, *output_pixels_obj; \
     \
    /* Parse the input tuple */ \
    if (!PyArg_ParseTuple(args, "IIOO", &width, &height, \
        &input_pixels_obj, &output_pixels_obj)) \
        return NULL; \
     \
    /* Interpret the input objects as numpy arrays. */ \
    PyObject *input_pixels_array = PyArray_FROM_OTF(input_pixels_obj, input_numpy_type, NPY_IN_ARRAY); \
    PyObject *output_pixels_array = PyArray_FROM_OTF(output_pixels_obj, output_numpy_type, NPY_INOUT_ARRAY); \
     \
    /* If that didn't work, throw an exception. */ \
    if (input_pixels_array == NULL || output_pixels_array == NULL) { \
        Py_XDECREF(input_pixels_array); \
        Py_XDECREF(output_pixels_array); \
        return NULL; \
    } \
     \
    /* Get pointers to the data as C-types. */ \
    input_c_type *input_pixels = (input_c_type*)PyArray_DATA(input_pixels_array); \
    output_c_type *output_pixels = (output_c_type*)PyArray_DATA(output_pixels_array); \
     \
    const bool status = func_name(width, height, input_pixels, output_pixels); \
     \
    /* Clean up. */ \
    Py_DECREF(input_pixels_array); \
    Py_DECREF(output_pixels_array); \
     \
    /* Build the output */ \
    return Py_BuildValue("O", status ? Py_True : Py_False); \
}

DEFINE_CONVERSION_WRAPPER(r8_to_rgba8, NPY_UINT8, unsigned char, NPY_UINT8, unsigned char)

DEFINE_CONVERSION_WRAPPER(rgb565_to_rgba8, NPY_UINT16, unsigned short, NPY_UINT8, unsigned char)
DEFINE_CONVERSION_WRAPPER(rgb8_to_rgba8, NPY_UINT8, unsigned char, NPY_UINT8, unsigned char)
DEFINE_CONVERSION_WRAPPER(rgba8_to_rgb8, NPY_UINT8, unsigned char, NPY_UINT8, unsigned char)

DEFINE_CONVERSION_WRAPPER(rgba4_to_rgba8, NPY_UINT16, unsigned short, NPY_UINT8, unsigned char)
DEFINE_CONVERSION_WRAPPER(rgba5551_to_rgba8, NPY_UINT16, unsigned short, NPY_UINT8, unsigned char)

DEFINE_CONVERSION_WRAPPER(luminance_alpha_to_rgba8, NPY_UINT8, unsigned char, NPY_UINT8, unsigned char)
DEFINE_CONVERSION_WRAPPER(luminance_to_rgba8, NPY_UINT8, unsigned char, NPY_UINT8, unsigned char)
DEFINE_CONVERSION_WRAPPER(alpha_to_rgba8, NPY_UINT8, unsigned char, NPY_UINT8, unsigned char)

static PyMethodDef module_methods[] = {
    {"r8_to_rgba8", glimage_r8_to_rgba8, METH_VARARGS, "Convert pixels from GL_R8 to GL_RGBA8"},

    {"rgb565_to_rgba8", glimage_rgb565_to_rgba8, METH_VARARGS, "Convert pixels from GL_RGB565 to GL_RGBA8"},
    {"rgb8_to_rgba8", glimage_rgb8_to_rgba8, METH_VARARGS, "Convert pixels from GL_RGB8 to GL_RGBA8"},
    {"rgba8_to_rgb8", glimage_rgba8_to_rgb8, METH_VARARGS, "Convert pixels from GL_RGBA8 to GL_RGB8"},

    {"rgba4_to_rgba8", glimage_rgba4_to_rgba8, METH_VARARGS, "Convert pixels from GL_RGBA4 to GL_RGBA8"},
    {"rgba5551_to_rgba8", glimage_rgba5551_to_rgba8, METH_VARARGS, "Convert pixels from GL_RGBA5551 to GL_RGBA8"},
    {"luminance_alpha_to_rgba8", glimage_luminance_alpha_to_rgba8, METH_VARARGS, "Convert pixels from GL_LUMINANCE_ALPHA to GL_RGBA8"},
    {"alpha_to_rgba8", glimage_alpha_to_rgba8, METH_VARARGS, "Convert pixels from GL_ALPHA to GL_RGBA8"},
    {"luminance_to_rgba8", glimage_luminance_to_rgba8, METH_VARARGS, "Convert pixels from GL_LUMINANCE to GL_RGBA8"},

    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_glimage(void)
{
    PyObject *m = Py_InitModule3("_glimage", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}
