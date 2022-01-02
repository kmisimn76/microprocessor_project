typedef unsigned char  BYTE;
typedef unsigned short WORD;
typedef unsigned long  DWORD;
typedef long           LONG;

 

typedef struct _BITMAPFILEHEADER {
    WORD    bfType;
    DWORD   bfSize;
    WORD    bfReserved1;
    WORD    bfReserved2;
    DWORD   bfOffBits;
    } BITMAPFILEHEADER;

 

typedef struct _BITMAPINFOHEADER {
    DWORD  biSize;
    LONG   biWidth;
    LONG   biHeight;
    WORD   biPlanes;
    WORD   biBitCount;
    DWORD  biCompression;
    DWORD  biSizeImage;
    LONG   biXPelsPerMeter;
    LONG   biYPelsPerMeter;
    DWORD  biClrUsed;
    DWORD  biClrImportant;
    } BITMAPINFOHEADER;

 


unsigned char *read_bmp(char *filename, BITMAPINFOHEADER *bmpHeader);


#include <stdio.h>
#include <stdlib.h>
#include <string.h>



unsigned char *
read_bmp(char *filename, BITMAPINFOHEADER *bmpHeader)
{
    FILE *filePtr; //our file pointer
    unsigned char *bitmapImage;  //store image data
    int imageIdx=0;  //image index counter
    unsigned char tempRGB;  //our swap variable
 BITMAPFILEHEADER bmfh;
 BITMAPINFOHEADER bmih;


    //open filename in read binary mode
    filePtr = fopen(filename,"rb");
    if (filePtr == NULL)
        return NULL;
    //read the bitmap file header
    
    fread(&bmfh, 14, 1, filePtr);  // 파일헤더 읽어들임
    fread(bmpHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);  // 정보헤더 읽어들임    //move file point to the begging of bitmap data

    //allocate enough memory for the bitmap image data
    bitmapImage = (unsigned char*)malloc(bmpHeader->biSizeImage);
    //verify memory allocation
    if (!bitmapImage)
    {
        free(bitmapImage);
        fclose(filePtr);
        return NULL;
    }
    //read in the bitmap image data
    fread(bitmapImage, bmpHeader->biSizeImage,1,filePtr);

    //make sure bitmap image data was read
    if (bitmapImage == NULL)
    {
        fclose(filePtr);
        return NULL;
    }

    //close file and return bitmap iamge data
    fclose(filePtr);
    return bitmapImage;
}


