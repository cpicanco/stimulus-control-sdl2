{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.video.writer.windows;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Windows, SDL2, Graphics.BMPtoPNG;

type

  TFrame = record
    Timestamp : Extended;
    InMemoryDeviceContext : HDC;
    WindowDeviceContext : HDC;
    Window  : HWND;
    Width : LongInt;
    Height : LongInt;
  end;

  { TVideoWriter }

  TVideoWriter = class(TThread)
  private
    FMsg: string;
    FFrame : TFrame;
    procedure PrintMessage;
    procedure UpdateWindowSurface;
  protected
    procedure Execute; override;
  public
    constructor Create(ARect : TSDL_Rect);
    destructor Destroy; override;
    procedure Log(const AMsg: string; AppendLineEnd: boolean = true);
    property Msg : string read FMsg write FMsg;
    procedure StartRecording;
    procedure Stop;
    procedure MainThreadSynchronize;
  end;

  var
    StartTimestamp : Extended;
    VideoWriter  : TVideoWriter;

implementation

uses
  sdl2_image
  //, pixels.format.conversion
  , avutil, avcodec, avformat, swscale
  , timestamps.methods
  , sdl.app.video.methods
  //, sdl.app.output
  ;

var
  ACriticalSection : TRTLCriticalSection;

{ TVideoWriter }

procedure TVideoWriter.PrintMessage;
begin
  WriteLn(Msg);
end;

procedure TVideoWriter.UpdateWindowSurface;
begin

end;

procedure TVideoWriter.Execute;
var
  LConversor : TBMPtoPNGConversor;
  LStart  : Int64;
  LEnd    : Int64;

  videoCodec: PAVCodec = nil;
  videoStream: PAVStream;
  codecParameters : PAVCodecParameters;
  outputFormat : AVOutputFormat;
  codecContext: PAVCodecContext;
  formatContext: PAVFormatContext;
  image: PSDL_Surface;
  packet: AVPacket;
  frame: PAVFrame;
  filename: string;
  frame_pts: Int64;
  fps: Double;
  opts : PAVDictionary;
  CodecName : PAnsiChar = 'MJPEG';
  res : integer;

  function Elapsed : string;
  begin
    Result := 'recording'+DirectorySeparator+FloatToStrF(
      FFrame.Timestamp - StartTimestamp, ffFixed, 0, 9)+'.png';
  end;
  // http://www.dranger.com/ffmpeg/tutorial02.html
  // https://ffmpeg.org/doxygen/3.1/structAVFrame.html
begin
  NameThreadForDebugging(ClassName);
  filename := 'output.mp4';
  fps := 30.0; // Set your desired frame rate
  av_register_all;

  //videoCodec := av_codec_next(videoCodec);
  //while videoCodec <> nil do begin
  //  WriteStr(filename, Integer(videoCodec^.id));
  //  WriteLn(filename, #32, StrPas(videoCodec^.long_name));
  //  videoCodec := av_codec_next(videoCodec);
  //end;

  videoCodec := avcodec_find_encoder(AV_CODEC_ID_MPEG4); // Use your desired codec
  //videoCodec := avcodec_find_encoder_by_name(CodecName);
  if not Assigned(videoCodec) then begin
    Terminate;
    Exit;
  end;

  outputFormat.long_name := videoCodec^.long_name;
  outputFormat.name := videoCodec^.name;
  outputFormat.video_codec := videoCodec^.id;

  if avformat_alloc_output_context2(
    @formatContext, @outputFormat, nil, PAnsiChar(filename)) < 0 then
  begin
    WriteLn('Could not create AVFormatContext');
    Terminate;
    Exit;
  end;
  //outputFormat := formatContext^.oformat;
  videoStream := avformat_new_stream(formatContext, videoCodec);
  videoStream^.time_base.num := 1;
  videoStream^.time_base.den := 30;
  videoStream^.avg_frame_rate.num := 1;
  videoStream^.avg_frame_rate.den := 30;
  videoStream^.codecpar := avcodec_parameters_alloc;

  //avcodec_parameters_from_context(videoStream^.codecpar, codecContext);
  videoStream^.codecpar^.bit_rate := 128000 {400000};
  videoStream^.codecpar^.width := WindowSize.w;
  videoStream^.codecpar^.height := WindowSize.h;
  videoStream^.codecpar^.codec_id := videoCodec^.id;
  videoStream^.codecpar^.codec_type := videoCodec^.type_;

  codecContext := avcodec_alloc_context3(videoCodec);
  codecContext^.bit_rate := 400000;
  codecContext^.width := WindowSize.w;;
  codecContext^.height := WindowSize.h;
  codecContext^.pkt_timebase.num := videoStream^.time_base.num;
  codecContext^.pkt_timebase.den := videoStream^.time_base.den;
  codecContext^.time_base.num := videoStream^.time_base.num;
  codecContext^.time_base.den := videoStream^.time_base.den;
  codecContext^.framerate.num := videoStream^.avg_frame_rate.num;
  codecContext^.framerate.den := videoStream^.avg_frame_rate.den;

  //codecContext^.codec^.p := ;
  avcodec_parameters_from_context(videoStream^.codecpar, codecContext);

  av_dump_format(formatContext, 0, @StdOut, 1);
  //if avcodec_open2(codecContext, codecContext^.codec, nil) < 0 then begin
  //  Exit;
  //end;

  if avio_open(@formatContext^.pb,
     PAnsiChar(filename), AVIO_FLAG_WRITE) < 0 then begin
    Exit;
  end;


  avformat_write_header(formatContext, nil);

  frame := av_frame_alloc;
  av_init_packet(@packet);

  WriteLn('|SDL_RenderReadPixels|IMG_SavePNG|');
  try
    while not Terminated do begin
      //av_frame_unref(frame);
      Synchronize(@UpdateWindowSurface);

      LStart := SDL_GetTicks64;
      SDL_RenderReadPixels(
        PSDLRenderer, nil, 0, PSDLSurface^.pixels, PSDLSurface^.pitch);

      frame^.pts := ET.GetSystemTicks;
      case PSDLSurface^.format^.format of
        SDL_PIXELFORMAT_RGB888 : begin
          //WriteLn('SDL_PIXELFORMAT_RGB888');
          //frame^.format := LongInt(AV_PIX_FMT_ARGB);
          //frame^.width := PSDLSurface^.w;
          //frame^.height := PSDLSurface^.h;
          //frame^.linesize[0] := PSDLSurface^.pitch;
          //frame^.linesize[1] := 0;
          //frame^.linesize[2] := 0;
          //frame^.data[0] := PSDLSurface^.pixels;
          //frame^.data[1] := nil;
          //frame^.data[2] := nil;
          WriteLn('SDL_PIXELFORMAT_RGB888');
          frame^.format := LongInt(AV_PIX_FMT_YUV420P);
          frame^.width := PSDLSurface^.w;
          frame^.height := PSDLSurface^.h;
          frame^.linesize[0] := PSDLSurface^.pitch;
          frame^.linesize[1] := PSDLSurface^.pitch;
          frame^.linesize[2] := PSDLSurface^.pitch;
          frame^.data[0] := PSDLSurface^.pixels;
          frame^.data[1] := PSDLSurface^.pixels;
          frame^.data[2] := PSDLSurface^.pixels;
          frame^.data[3] := nil;
          frame^.data[4] := nil;
          frame^.data[5] := nil;
          frame^.data[6] := nil;
          frame^.data[7] := nil;
        end;
        else { implement me }
      end;

      res := avcodec_send_frame(codecContext, frame);
      writeln(res.ToString);
      //case res of
      //  //EAGAIN :;
      //  AVERROR_EOF: ;
      //  //AVERROR(EINVAL):;
      //  //AVERROR(ENOMEM):;
      //  else { enconding errors } WriteLn(res);
      //
      //end;


      packet.data := nil;
      packet.size := 0;

      if avcodec_receive_packet(codecContext, @packet) < 0 then
        Continue;

      av_interleaved_write_frame(formatContext, @packet);
      //av_write_frame(formatContext, @packet);

      LEnd := SDL_GetTicks64;
      Write('|'+((LEnd - LStart)*1e-3).ToStringF);

      LStart := SDL_GetTicks64;
      //IMG_SavePNG(PSDLSurface, PAnsiChar(Elapsed));

      LEnd := SDL_GetTicks64;
      WriteLn('|'+((LEnd - LStart)*1e-3).ToStringF+'|');
    end;
  finally
    // Finalize the video encoding, close the container, and release resources.
    av_write_trailer(@formatContext);
    av_packet_unref(@packet);
    av_frame_free(@frame);
    avcodec_parameters_free(@videoStream^.codecpar);
    avcodec_close(codecContext);
    avformat_free_context(formatContext);
  end;
  Log(ClassName+': Terminated ...');
end;

constructor TVideoWriter.Create(ARect : TSDL_Rect);
begin
  inherited Create(False);
  //Priority := tpTimeCritical;
  FreeOnTerminate := True;

  with FFrame do begin
    Timestamp := 0;
    Width := ARect.w;
    Height := ARect.h;
    WindowDeviceContext := WindowDeviceContextHandle;
    Window  := WindowHandle;
    InMemoryDeviceContext := CreateCompatibleDC(WindowDeviceContext);
  end;
end;

destructor TVideoWriter.Destroy;
begin
  DeleteDC(FFrame.InMemoryDeviceContext);
  inherited Destroy;
end;

procedure TVideoWriter.Stop;
begin
  Terminate;
end;

procedure TVideoWriter.MainThreadSynchronize;
begin
  // console applications requires CheckSynchronize from the main periodically
  // CheckSynchronize;
end;

procedure TVideoWriter.Log(const AMsg: string; AppendLineEnd: boolean);
var
  s: String;
begin
  EnterCriticalsection(ACriticalSection);
  s:=AMsg;
  if AppendLineEnd then
    s:=s+LineEnding;
  Msg:=s;
  Synchronize(@PrintMessage);
  LeaveCriticalsection(ACriticalSection);
end;

procedure TVideoWriter.StartRecording;
begin
  Sleep(50);
  Start;
end;


initialization
  StartTimestamp := 0;
  InitCriticalSection(ACriticalSection);

finalization
  DoneCriticalsection(ACriticalSection);

end.

