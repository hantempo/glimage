#include "glimage.h"

bool r8_to_rgba8(unsigned int width, unsigned int height, unsigned char *input_pixels, unsigned char *output_pixels)
{
    for (unsigned int index = 0; index < width * height; ++index)
    {
        *(output_pixels++) = *(input_pixels++);
        *(output_pixels++) = 0;
        *(output_pixels++) = 0;
        *(output_pixels++) = 0xFF;
    }
    return true;
}

bool rgb565_to_rgba8(unsigned int width, unsigned int height, unsigned short *input_pixels, unsigned char *output_pixels)
{
    for (unsigned int index = 0; index < width * height; ++index)
    {
        const unsigned short pixel = *(input_pixels++);
        *(output_pixels++) = ((pixel >> 11) & 0x1F) * 0xFF / 0x1F;
        *(output_pixels++) = ((pixel >> 5) & 0x3F) * 0xFF / 0x3F;
        *(output_pixels++) = (pixel & 0x1F) * 0xFF / 0x1F;
        *(output_pixels++) = 0xFF;
    }
    return true;
}

bool rgba4_to_rgba8(unsigned int width, unsigned int height, unsigned short *input_pixels, unsigned char *output_pixels)
{
    for (unsigned int index = 0; index < width * height; ++index)
    {
        const unsigned short pixel = *(input_pixels++);
        *(output_pixels++) = ((pixel >> 12) & 0x0F) * 0xFF / 0x0F;
        *(output_pixels++) = ((pixel >> 8) & 0x0F) * 0xFF / 0x0F;
        *(output_pixels++) = ((pixel >> 4) & 0x0F) * 0xFF / 0x0F;
        *(output_pixels++) = (pixel & 0x0F) * 0xFF / 0x0F;
    }
    return true;
}

bool rgba5551_to_rgba8(unsigned int width, unsigned int height, unsigned short *input_pixels, unsigned char *output_pixels)
{
    for (unsigned int index = 0; index < width * height; ++index)
    {
        const unsigned short pixel = *(input_pixels++);
        *(output_pixels++) = ((pixel >> 11) & 0x1F) * 0xFF / 0x1F;
        *(output_pixels++) = ((pixel >> 6) & 0x1F) * 0xFF / 0x1F;
        *(output_pixels++) = ((pixel >> 1) & 0x1F) * 0xFF / 0x1F;
        *(output_pixels++) = (pixel & 0x01) * 0xFF / 0x01;
    }
    return true;
}

bool luminance_alpha_to_rgba8(unsigned int width, unsigned int height, unsigned char *input_pixels, unsigned char *output_pixels)
{
    for (unsigned int index = 0; index < width * height; ++index)
    {
        const unsigned char luminance = *(input_pixels++);
        const unsigned char alpha = *(input_pixels++);
        *(output_pixels++) = luminance;
        *(output_pixels++) = luminance;
        *(output_pixels++) = luminance;
        *(output_pixels++) = alpha;
    }
    return true;
}

bool luminance_to_rgba8(unsigned int width, unsigned int height, unsigned char *input_pixels, unsigned char *output_pixels)
{
    for (unsigned int index = 0; index < width * height; ++index)
    {
        const unsigned char luminance = *(input_pixels++);
        *(output_pixels++) = luminance;
        *(output_pixels++) = luminance;
        *(output_pixels++) = luminance;
        *(output_pixels++) = 0xFF;
    }
    return true;
}

bool alpha_to_rgba8(unsigned int width, unsigned int height, unsigned char *input_pixels, unsigned char *output_pixels)
{
    for (unsigned int index = 0; index < width * height; ++index)
    {
        const unsigned char alpha = *(input_pixels++);
        *(output_pixels++) = 0;
        *(output_pixels++) = 0;
        *(output_pixels++) = 0;
        *(output_pixels++) = alpha;
    }
    return true;
}