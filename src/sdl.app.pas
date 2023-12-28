{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , ctypes
  , sdl.app.video.types
  , sdl.app.system.keyboard
  {$IFDEF NO_LCL}
  , sdl.app.events.nolcl
  {$ELSE}
  , sdl.app.events.custom
  , sdl.app.events.abstract
  {$ENDIF}
  ;

type

  { TSDLApplication }

  TSDLApplication = class
    private
      FKeyboard : TSDLSystemKeyboard;
      FTitle : PAnsichar;
      FCurrentMonitorIndex : cint;
      FOnClose: TNotifyEvent;
      FRunning: Boolean;
      FSDLWindow: PSDL_Window;
      FSDLRenderer: PSDL_Renderer;
      FSDLSurface : PSDL_Surface;
      FMonitors : TMonitors;
      function GetCurrentMonitor : TSDL_Rect;
      function GetEvents: TCustomEventHandler;
      function GetMonitor(i : cint): TSDL_Rect;
      procedure SetCurrentMonitor(i : cint);
      procedure SetOnClose(AValue: TNotifyEvent);
    {$IFNDEF NO_LCL}
    private
      FOnKeyDown: TOnKeyDownEvent;
      function GetShowMarkers: Boolean;
      procedure SetOnKeyDown(AValue: TOnKeyDownEvent);
      procedure SetShowMarkers(AValue: Boolean);
    public
      procedure SetupAudio;
      procedure SetupText;
      property ShowMarkers : Boolean read GetShowMarkers write SetShowMarkers;
    {$ENDIF}
    public
      class procedure LoadMonitors(var AMonitors: TMonitors);
      class procedure GetAvailableMonitors(AStrings: TStrings);
      constructor Create(ATitle : PAnsiChar = 'Stimulus Control'); reintroduce;
      destructor Destroy; override;
      procedure Run;
      procedure SetupVideo(AMonitor : cint = 0);
      procedure Terminate;
      property Running : Boolean read FRunning write FRunning;
      property Window  : PSDL_Window read FSDLWindow;
      property Monitor : TSDL_Rect read GetCurrentMonitor;
      property Events  : TCustomEventHandler read GetEvents;
      property OnClose : TNotifyEvent read FOnClose write SetOnClose;
  end;

var
  SDLApp : TSDLApplication;

implementation

uses sdl2_image
  , sdl.app.output
  , sdl.app.controller.manager
  , sdl.app.video.methods
  , sdl.app.mouse
{$IFDEF NO_LCL}
  , sdl.app.renderer.nolcl
{$ELSE}
  , sdl.app.markers
  , sdl.app.text
  , sdl.app.audio
  , sdl.app.renderer
{$ENDIF}
  ;

{ TSDLApplication }

class procedure TSDLApplication.LoadMonitors(var AMonitors: TMonitors);
var
  i: Integer;
begin
  SetLength(AMonitors, SDL_GetNumVideoDisplays);
  for i := Low(AMonitors) to High(AMonitors) do begin
    SDL_GetDisplayBounds(i, @AMonitors[i]);
  end;
end;

procedure TSDLApplication.SetOnClose(AValue: TNotifyEvent);
begin
  if FOnClose=AValue then Exit;
  FOnClose:=AValue;
end;

{$IFNDEF NO_LCL}
procedure TSDLApplication.SetShowMarkers(AValue: Boolean);
begin
  if ShowMarkers = AValue then Exit;
  if AValue then begin
    Markers := TMarkers.Create;
    Markers.LoadFromFile;
  end else begin
    FreeAndNil(Markers);
  end;
end;
{$ENDIF}

{$IFNDEF NO_LCL}
function TSDLApplication.GetShowMarkers: Boolean;
begin
  Result := Assigned(Markers);
end;

procedure TSDLApplication.SetOnKeyDown(AValue: TOnKeyDownEvent);
begin
  if FOnKeyDown = AValue then Exit;
  FOnKeyDown := AValue;
end;

{$ENDIF}

procedure TSDLApplication.SetCurrentMonitor(i: cint);
begin
  LoadMonitors(FMonitors);
  if i > SDL_GetNumVideoDisplays then Exit;
  with FMonitors[i] do begin
    SDL_SetWindowPosition(FSDLWindow, x, y);
    SDL_SetWindowSize(FSDLWindow, w, h);
  end;
end;

function TSDLApplication.GetCurrentMonitor: TSDL_Rect;
begin
  Result := FMonitors[FCurrentMonitorIndex];
end;

function TSDLApplication.GetEvents: TCustomEventHandler;
begin
  Result := SDLEvents;
end;

function TSDLApplication.GetMonitor(i: cint): TSDL_Rect;
begin
  LoadMonitors(FMonitors);
  if i > SDL_GetNumVideoDisplays then Exit;
  Result := FMonitors[i];
end;

constructor TSDLApplication.Create(ATitle: PAnsiChar);
var
  LError : string;
begin
  Print(Self.ClassName+'.'+{$I %CURRENTROUTINE%}+#32+ATitle);
  FTitle := ATitle;
  SDLEvents := TCustomEventHandler.Create;

  // errors in the video subsystem may be related to
  // SDL_VIDEODRIVER system environment variable (particularly on windows)
  // https://wiki.libsdl.org/SDL2/FAQUsingSDL
  if SDL_InitSubSystem(SDL_INIT_VIDEO) < 0 then begin
    LError := SDL_GetError;
    Print(LError);
    raise Exception.Create(LError);
  end;
  LoadMonitors(FMonitors);

  if SDL_InitSubSystem(SDL_INIT_TIMER) < 0 then begin
    LError := SDL_GetError;
    Print(LError);
    raise Exception.Create(LError);
  end;

  IMG_Init(IMG_INIT_PNG);

  {$IFNDEF NO_LCL}
  if SDL_InitSubSystem(SDL_INIT_AUDIO) < 0 then begin
    LError := SDL_GetError;
    Print(LError);
    raise Exception.Create(LError);
  end;
  SDLAudio := TSDLAudio.Create;

  if SDL_InitSubSystem(SDL_INIT_GAMECONTROLLER) < 0  then begin
    LError := SDL_GetError;
    Print(LError);
    raise Exception.Create(LError);
  end;

  // text/font setup
  SDLText  := TSDLText.Create;
  {$ENDIF}
end;

destructor TSDLApplication.Destroy;
begin
  {$IFNDEF NO_LCL}
  if Assigned(Markers) then begin
    Markers.Free;
  end;
  {$ENDIF}
  inherited Destroy;
end;

procedure TSDLApplication.Run;
begin
  FRunning:=True;
  try
    while FRunning do begin
      SDLEvents.HandlePending;
      Render;
    end;
  finally
    {$IFNDEF NO_LCL}
    if Assigned(SDLText) then begin
      SDLText.Free;
      SDLText := nil;
    end;

    if Assigned(SDLAudio) then begin
      SDLAudio.Free;
      SDLAudio := nil;
    end;
    {$ENDIF}

    if Assigned(SDLEvents) then begin
      SDLEvents.Free;
      //SdlEvents := nil;
    end;

    if Assigned(Mouse) then begin
      Mouse.Free;
      Mouse := nil;
    end;

    SDL_DestroyRenderer(FSDLRenderer);
    SDL_DestroyWindow(FSDLWindow);
    SDL_Quit;
  end;

  Print('Good Bye');
  if Assigned(OnClose) then
    OnClose(Self);
end;

class procedure TSDLApplication.GetAvailableMonitors(AStrings: TStrings);
var
  i: Integer;
  LMonitors : array of TSDL_Rect;
  LError: PAnsiChar;
begin
  if SDL_InitSubSystem(SDL_INIT_VIDEO) < 0 then begin
    LError := SDL_GetError;
    Print(LError);
    raise Exception.Create(LError);
  end;
  LoadMonitors(LMonitors);

  AStrings.Clear;
  for i := Low(LMonitors) to High(LMonitors) do begin
    AStrings.Append('Monitor ' + String.Join(#32,
      [i, '-', LMonitors[i].x, LMonitors[i].y, LMonitors[i].w, LMonitors[i].h]));
  end;
  SDL_Quit;
end;

procedure TSDLApplication.SetupVideo(AMonitor: cint);
var
  LMonitor : TSDL_Rect;
begin
  if AMonitor > SDL_GetNumVideoDisplays then Exit;

  FCurrentMonitorIndex := AMonitor;

  LMonitor := FMonitors[AMonitor];
  FSDLWindow := SDL_CreateWindow(FTitle, LMonitor.x, LMonitor.y,
    LMonitor.w, LMonitor.h, 0);
  Mouse := TSDLMouseHandler.Create(FSDLWindow);
  FSDLRenderer := SDL_CreateRenderer(FSDLWindow, -1, SDL_RENDERER_ACCELERATED);
  FSDLSurface  := SDL_GetWindowSurface(FSDLWindow);
  AssignVariables(FSDLWindow, FSDLRenderer, FSDLSurface);

  {$IFDEF NO_LCL}
  sdl.app.renderer.nolcl.Monitor := Self.Monitor;
  {$ELSE}

  {$ENDIF}
end;

procedure TSDLApplication.Terminate;
begin
  FRunning := False;
  Print('Good bye');
end;

{$IFNDEF NO_LCL}
procedure TSDLApplication.SetupAudio;
begin
  AllocateDefaultAudioChannels;
end;

procedure TSDLApplication.SetupText;
begin
  SDLText.SetupFonts;
end;
{$ENDIF}

end.

