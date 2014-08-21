#include <Python.h>
#include <numpy/arrayobject.h>
#include "glimage.h"

static char module_docstring[] =
    "This module provides some methods to convert pixels of images between different OpenGL formats.";

static PyObject *glimage_rgba4_to_rgba8(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"rgba4_to_rgba8", glimage_rgba4_to_rgba8, METH_VARARGS, "Convert pixels from GL_RGBA4 to GL_RGBA8"},
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

static PyObject *glimage_rgba4_to_rgba8(PyObject *self, PyObject *args)
{
    unsigned int width, height;
    PyObject *input_pixels_obj, *output_pixels_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "IIOO!", &width, &height,
        &input_pixels_obj, &PyArray_Type, &output_pixels_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *input_pixels_array = PyArray_FROM_OTF(input_pixels_obj, NPY_UINT16, NPY_IN_ARRAY);
    PyObject *output_pixels_array = PyArray_FROM_OTF(output_pixels_obj, NPY_UINT8, NPY_INOUT_ARRAY);

    /* If that didn't work, throw an exception. */
    if (input_pixels_array == NULL || output_pixels_array == NULL) {
        Py_XDECREF(input_pixels_array);
        Py_XDECREF(output_pixels_array);
        return NULL;
    }
    
    /* Get pointers to the data as C-types. */
    unsigned short *input_pixels = (unsigned short*)PyArray_DATA(input_pixels_array);
    unsigned char *output_pixels = (unsigned char*)PyArray_DATA(output_pixels_array);

    const bool status = rgba4_to_rgba8(width, height, input_pixels, output_pixels);

    /* Clean up. */
    Py_DECREF(input_pixels_array);
    Py_DECREF(output_pixels_array);
    Py_INCREF(Py_None);
    
    /* Build the output */
    return Py_BuildValue("O", status ? Py_True : Py_False);
}