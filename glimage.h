// GL_RED
bool r8_to_rgba8(unsigned int width, unsigned int height,
    unsigned char *input_pixels, unsigned char *output_pixels);

// GL_RGB
bool rgb565_to_rgba8(unsigned int width, unsigned int height,
    unsigned short *input_pixels, unsigned char *output_pixels);
bool rgb8_to_rgba8(unsigned int width, unsigned int height,
    unsigned char *input_pixels, unsigned char *output_pixels);
bool rgba8_to_rgb8(unsigned int width, unsigned int height,
    unsigned char *input_pixels, unsigned char *output_pixels);

// GL_RGBA
bool rgba4_to_rgba8(unsigned int width, unsigned int height,
    unsigned short *input_pixels, unsigned char *output_pixels);
bool rgba5551_to_rgba8(unsigned int width, unsigned int height,
    unsigned short *input_pixels, unsigned char *output_pixels);

// GL_LUMINANCE, GL_ALPHA & GL_LUMINANCE_ALPHA
bool luminance_alpha_to_rgba8(unsigned int width, unsigned int height,
    unsigned char *input_pixels, unsigned char *output_pixels);
bool alpha_to_rgba8(unsigned int width, unsigned int height,
    unsigned char *input_pixels, unsigned char *output_pixels);
bool luminance_to_rgba8(unsigned int width, unsigned int height,
    unsigned char *input_pixels, unsigned char *output_pixels);
