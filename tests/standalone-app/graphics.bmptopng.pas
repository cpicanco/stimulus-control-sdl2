unit Graphics.BMPtoPNG;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Graphics, IntfGraphics, FPImage, FPWritePNG, FPReadBMP;

type

  { TBMPtoJPEGConversor }

  TBMPtoPNGConversor = class(TBitmap)
  protected
    class function GetWriterClass: TFPCustomImageWriterClass; override;
    procedure InitializeWriter(AImage: TLazIntfImage; AWriter: TFPCustomImageWriter); override;
  end;

implementation

uses ZStream;

{ TBMPtoJPEGConversor }

class function TBMPtoPNGConversor.GetWriterClass: TFPCustomImageWriterClass;
begin
  Result := TFPWriterPNG;
end;

procedure TBMPtoPNGConversor.InitializeWriter(AImage: TLazIntfImage;
  AWriter: TFPCustomImageWriter);
var
  LFPWriterPNG : TFPWriterPNG;
begin
  inherited InitializeWriter(AImage, AWriter);
  LFPWriterPNG := AWriter as TFPWriterPNG;
  LFPWriterPNG.CompressionLevel := clmax;
  LFPWriterPNG.GrayScale := False;
  LFPWriterPNG.UseAlpha := False;
end;

end.
