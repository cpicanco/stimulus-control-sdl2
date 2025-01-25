{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.video.writer;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, sdl2, sdl2_image;

type

  TFrame = record
    Timestamp : Extended;
    Surface   : PSDL_Surface;
  end;

  { TVideoWriter }

  TVideoWriter = class(TThread)
  private
    FRTLEventMainThread : PRTLevent;
    FMsg: string;
    FFrame : TFrame;
    procedure SaveToJPEG;
    procedure FreeFrame;
    procedure PrintMessage;
  protected
    procedure Execute; override;
  public
    constructor Create(ARect : TSDL_Rect);
    destructor Destroy; override;
    procedure Log(const AMsg: string; AppendLineEnd: boolean = true);
    property Msg : string read FMsg write FMsg;
    procedure RecordFrame;
    procedure Stop;
  end;

  var
    StartTimestamp : Extended;
    VideoWriter  : TVideoWriter;

implementation

uses
  timestamps.methods, sdl.app.video.methods
  //, sdl.app.output
  ;

var
  ACriticalSection : TRTLCriticalSection;

{ TVideoWriter }

procedure TVideoWriter.SaveToJPEG;
  function Elapsed : PAnsiChar;
  begin
    Result := PAnsiChar(FloatToStrF(
      FFrame.Timestamp - StartTimestamp, ffFixed, 0, 9)+'.jpeg');
  end;
begin
  EnterCriticalSection(ACriticalSection);
  IMG_SaveJPG(FFrame.Surface, Elapsed, 100); // 100 is the quality level
  LeaveCriticalSection(ACriticalSection);
end;

procedure TVideoWriter.FreeFrame;
begin
  SDL_FreeSurface(FFrame.Surface);
  FFrame.Surface := nil;
  FFrame.Timestamp := 0;
end;

procedure TVideoWriter.PrintMessage;
begin
  WriteLn(Msg);
end;

procedure TVideoWriter.Execute;
var
  LEnd   : Extended;
begin
  NameThreadForDebugging(ClassName);
  FRTLEventMainThread := RTLEventCreate;
  while not Terminated do begin
    RTLEventWaitFor(FRTLEventMainThread);
    LEnd := ET.Elapsed;
    WriteLn('Read elapsed '+ (LEnd - FFrame.Timestamp).ToStringF);
    SaveToJPEG;
    WriteLn('Write elapsed '+ (ET.Elapsed - LEnd).ToStringF);
  end;
  Log(ClassName+': Terminated ...');
end;

constructor TVideoWriter.Create(ARect : TSDL_Rect);
begin
  inherited Create(False);
  FreeOnTerminate := True;

  FFrame.Timestamp := 0;
  FFrame.Surface := SDL_CreateRGBSurface(0, ARect.w, ARect.h,
    32, $00FF0000, $0000FF00, $000000FF, $FF000000);
  Start;
end;

destructor TVideoWriter.Destroy;
begin
  FreeFrame;
  inherited Destroy;
end;

procedure TVideoWriter.Stop;
begin
  RTLEventSetEvent(FRTLEventMainThread);
  RTLEventDestroy(FRTLEventMainThread);
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

procedure TVideoWriter.RecordFrame;
begin
  FFrame.Timestamp := ET.Elapsed;
  SDL_RenderReadPixels(PSDLRenderer, nil, SDL_PIXELFORMAT_ARGB8888,
    FFrame.Surface^.pixels, FFrame.Surface^.pitch);
  RTLEventSetEvent(FRTLEventMainThread);
  {$IFDEF NO_LCL}
  CheckSynchronize;
  {$ENDIF}
end;

initialization
  StartTimestamp := 0;
  InitCriticalSection(ACriticalSection);

finalization
  DoneCriticalsection(ACriticalSection);

end.

