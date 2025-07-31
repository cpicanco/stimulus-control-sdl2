unit pixels.format.conversion;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils;


procedure


procedure YUV420PToRGB888(const y : PByte; const u : PByte; const v : PByte;
  var rgb : PByte; width, height: Integer);

implementation

const
  MAX_BYTE = 256;

  function RNDTO2(X: Integer): Integer;
  begin
    Result := X and $FFFFFFFE;
  end;

  function RNDTO32(X: Integer): Integer;
  begin
    if (X mod 32) <> 0 then begin
      Result := (X + 32) and $FFFFFFE0
    end else begin
      Result := X;
    end;
  end;

  //procedure CloneSurfaceToFrame(surface: PSDL_Surface; frame: PAVFrame);
  //var
  //  row: Integer;
  //  srcPixels: PByte;
  //  dstData: PByte;
  //  srcPitch, dstLinesize: Integer;
  //begin
  //  // Check if the SDL surface format is compatible with the AVFrame format.
  //  case surface^.format^.format of
  //    SDL_PIXELFORMAT_UNKNOWN : WriteLn('SDL_PIXELFORMAT_UNKNOWN');
  //    SDL_PIXELFORMAT_INDEX1LSB : WriteLn('SDL_PIXELFORMAT_INDEX1LSB');
  //    SDL_PIXELFORMAT_INDEX1MSB : WriteLn('SDL_PIXELFORMAT_INDEX1MSB');
  //    SDL_PIXELFORMAT_INDEX4LSB : WriteLn('SDL_PIXELFORMAT_INDEX4LSB');
  //    SDL_PIXELFORMAT_INDEX4MSB : WriteLn('SDL_PIXELFORMAT_INDEX4MSB');
  //    SDL_PIXELFORMAT_INDEX8 : WriteLn('SDL_PIXELFORMAT_INDEX8');
  //    SDL_PIXELFORMAT_RGB332 : WriteLn('SDL_PIXELFORMAT_RGB332');
  //    SDL_PIXELFORMAT_RGB444 : WriteLn('SDL_PIXELFORMAT_RGB444');
  //    SDL_PIXELFORMAT_RGB555 : WriteLn('SDL_PIXELFORMAT_RGB555');
  //    SDL_PIXELFORMAT_BGR555 : WriteLn('SDL_PIXELFORMAT_BGR555');
  //    SDL_PIXELFORMAT_ARGB4444 : WriteLn('SDL_PIXELFORMAT_ARGB4444');
  //    SDL_PIXELFORMAT_RGBA4444 : WriteLn('SDL_PIXELFORMAT_RGBA4444');
  //    SDL_PIXELFORMAT_ABGR4444 : WriteLn('SDL_PIXELFORMAT_ABGR4444');
  //    SDL_PIXELFORMAT_BGRA4444 : WriteLn('SDL_PIXELFORMAT_BGRA4444');
  //    SDL_PIXELFORMAT_ARGB1555 : WriteLn('SDL_PIXELFORMAT_ARGB1555');
  //    SDL_PIXELFORMAT_RGBA5551 : WriteLn('SDL_PIXELFORMAT_RGBA5551');
  //    SDL_PIXELFORMAT_ABGR1555 : WriteLn('SDL_PIXELFORMAT_ABGR1555');
  //    SDL_PIXELFORMAT_BGRA5551 : WriteLn('SDL_PIXELFORMAT_BGRA5551');
  //    SDL_PIXELFORMAT_RGB565 : WriteLn('SDL_PIXELFORMAT_RGB565');
  //    SDL_PIXELFORMAT_BGR565 : WriteLn('SDL_PIXELFORMAT_BGR565');
  //    SDL_PIXELFORMAT_RGB24 : WriteLn('SDL_PIXELFORMAT_RGB24');
  //    SDL_PIXELFORMAT_BGR24 : WriteLn('SDL_PIXELFORMAT_BGR24');
  //    SDL_PIXELFORMAT_RGB888 : WriteLn('SDL_PIXELFORMAT_RGB888');
  //    SDL_PIXELFORMAT_RGBX8888 : WriteLn('SDL_PIXELFORMAT_RGBX8888');
  //    SDL_PIXELFORMAT_BGR888 : WriteLn('SDL_PIXELFORMAT_BGR888');
  //    SDL_PIXELFORMAT_BGRX8888 : WriteLn('SDL_PIXELFORMAT_BGRX8888');
  //    SDL_PIXELFORMAT_ARGB8888 : WriteLn('SDL_PIXELFORMAT_ARGB8888');
  //    SDL_PIXELFORMAT_RGBA8888 : WriteLn('SDL_PIXELFORMAT_RGBA8888');
  //    SDL_PIXELFORMAT_ABGR8888 : WriteLn('SDL_PIXELFORMAT_ABGR8888');
  //    SDL_PIXELFORMAT_BGRA8888 : WriteLn('SDL_PIXELFORMAT_BGRA8888');
  //    SDL_PIXELFORMAT_ARGB2101010 : WriteLn('SDL_PIXELFORMAT_ARGB2101010');
  //    SDL_PIXELFORMAT_YV12 : WriteLn('SDL_PIXELFORMAT_YV12');
  //    SDL_PIXELFORMAT_IYUV : WriteLn('SDL_PIXELFORMAT_IYUV');
  //    SDL_PIXELFORMAT_YUY2 : WriteLn('SDL_PIXELFORMAT_YUY2');
  //    SDL_PIXELFORMAT_UYVY : WriteLn('SDL_PIXELFORMAT_UYVY');
  //    SDL_PIXELFORMAT_YVYU : WriteLn('SDL_PIXELFORMAT_YVYU');
  //    SDL_PIXELFORMAT_NV12 : WriteLn('SDL_PIXELFORMAT_NV12');
  //    SDL_PIXELFORMAT_NV21 : WriteLn('SDL_PIXELFORMAT_NV21');
  //    SDL_PIXELFORMAT_EXTERMAL_OES : WriteLn('SDL_PIXELFORMAT_EXTERMAL_OES');
  //  end;
  //
  //  //if (surface^.w <> frame^.width) or
  //  //   (surface^.h <> frame^.height) then
  //  //begin
  //  //  Writeln('Incompatible formats or dimensions.');
  //  //  Exit;
  //  //end;
  //  //
  //  //// Calculate the size of a single row in bytes.
  //  //srcPitch := surface^.pitch;
  //  //dstLinesize := frame^.linesize[0];
  //  //
  //  //for row := 0 to frame^.height - 1 do
  //  //begin
  //  //  srcPixels := PByte(PtrUInt(surface^.pixels) + UInt32(row) * UInt32(srcPitch));
  //  //  dstData := frame^.data[0] + row * dstLinesize;
  //  //
  //  //  // Copy the pixel data from the SDL surface to the AVFrame.
  //  //  dstData^
  //  //  Move(srcPixels^, , dstLinesize);
  //  //end;
  //end;

function Clamp(value: Int16): Byte;
begin
  case value of
     Low(Int16)..-1        : Result := 0;
     MAX_BYTE..High(Int16) : Result := MAX_BYTE-1;
     else                    Result := Byte(Value);
  end;
end;

procedure YUV420PToRGB888(const y : PByte; const u : PByte; const v : PByte;
  var rgb : PByte; width, height: Integer);
var
  r, g, b, yy, uu, vv: Integer;
  i, j: Integer;
begin
  for j := 0 to height - 1 do begin
    for i := 0 to width - 1 do begin
      yy := y[j * width + i];
      uu := u[(j div 2) * (width div 2) + (i div 2)];
      vv := v[(j div 2) * (width div 2) + (i div 2)];

      r := Round(1.164 * (yy - 16) + 1.596 * (vv - 128));
      g := Round(1.164 * (yy - 16) - 0.813 * (vv - 128) - 0.391 * (uu - 128));
      b := Round(1.164 * (yy - 16) + 2.018 * (uu - 128));

      rgb^ := Clamp(r);
      Inc(rgb);
      rgb^ := Clamp(g);
      Inc(rgb);
      rgb^ := Clamp(b);
      Inc(rgb);
    end;
  end;
end;


end.

