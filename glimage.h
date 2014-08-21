bool r8_to_rgba8(unsigned int width, unsigned int height,
    unsigned char *input_pixels, unsigned char *output_pixels);

bool rgb565_to_rgba8(unsigned int width, unsigned int height,
    unsigned short *input_pixels, unsigned char *output_pixels);

bool rgba4_to_rgba8(unsigned int width, unsigned int height,
    unsigned short *input_pixels, unsigned char *output_pixels);

bool rgba5551_to_rgba8(unsigned int width, unsigned int height,
    unsigned short *input_pixels, unsigned char *output_pixels);

bool luminance_alpha_to_rgba8(unsigned int width, unsigned int height,
    unsigned char *input_pixels, unsigned char *output_pixels);

bool alpha_to_rgba8(unsigned int width, unsigned int height,
    unsigned char *input_pixels, unsigned char *output_pixels);

bool luminance_to_rgba8(unsigned int width, unsigned int height,
    unsigned char *input_pixels, unsigned char *output_pixels);