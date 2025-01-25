{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

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
      //FSDLGLContext : TSDL_GLContext;
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
      FShowMarkers: Boolean;
      procedure SetOnKeyDown(AValue: TOnKeyDownEvent);
      procedure SetShowMarkers(AValue: Boolean);
    public
      procedure SetupAudio;
      procedure SetupText;
      property ShowMarkers : Boolean read FShowMarkers write SetShowMarkers;
    {$ENDIF}
    public
      class procedure LoadMonitors(var AMonitors: TMonitors);
      class procedure GetAvailableMonitors(AStrings: TStrings);
      constructor Create(ATitle : PAnsiChar = 'Stimulus Control'); reintroduce;
      destructor Destroy; override;
      procedure Run;
      procedure PrintRendererSetup;
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
  //, GL, GLext
  //, sdl.app.output
  , sdl.app.video.methods
  , sdl.app.mouse
{$IFDEF NO_LCL}
  , sdl.app.renderer.nolcl
{$ELSE}
  , sdl.app.markers
  , sdl.app.text
  , sdl.app.audio
  , sdl.app.renderer
  , sdl.app.renderer.testmode
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
  if FShowMarkers = AValue then Exit;
  FShowMarkers := AValue;
end;
{$ENDIF}

{$IFNDEF NO_LCL}
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
  //Print(Self.ClassName+'.'+{$I %CURRENTROUTINE%}+#32+ATitle);
  FTitle := ATitle;
  SDLEvents := TCustomEventHandler.Create;

  // errors in the video subsystem may be related to
  // SDL_VIDEODRIVER system environment variable (particularly on windows)
  // https://wiki.libsdl.org/SDL2/FAQUsingSDL
  if SDL_InitSubSystem(SDL_INIT_VIDEO) < 0 then begin
    LError := SDL_GetError;
    //Print(LError);
    raise Exception.Create(LError);
  end;
  LoadMonitors(FMonitors);

  if SDL_InitSubSystem(SDL_INIT_TIMER) < 0 then begin
    LError := SDL_GetError;
    //Print(LError);
    raise Exception.Create(LError);
  end;

  IMG_Init(IMG_INIT_PNG);

  {$IFNDEF NO_LCL}
  if SDL_InitSubSystem(SDL_INIT_AUDIO) < 0 then begin
    LError := SDL_GetError;
    //Print(LError);
    raise Exception.Create(LError);
  end;
  SDLAudio := TSDLAudio.Create;

  if SDL_InitSubSystem(SDL_INIT_GAMECONTROLLER) < 0  then begin
    LError := SDL_GetError;
    //Print(LError);
    raise Exception.Create(LError);
  end;

  if SDL_Init(SDL_INIT_SENSOR) < 0 then
  begin
    LError := SDL_GetError;
    //Print(LError);
    raise Exception.Create(LError);
  end;

  // text/font setup
  SDLText  := TSDLText.Create;
  {$ENDIF}
end;

destructor TSDLApplication.Destroy;
begin

  inherited Destroy;
end;

procedure TSDLApplication.Run;
//type
//  TRenderMode = (rendNormal, rendTestMode);
//var
//  LRenderMode : TRenderMode = rendTestMode;
begin
  //case LRenderMode of
  //  rendNormal: { do nothing };
  //  rendTestMode: SDL.App.Renderer.TestMode.Initialize;
  //end;

  FRunning:=True;
  if FShowMarkers then begin
    Markers := TMarkers.Create;
    Markers.LoadFromFile;
  end;

  try
    while FRunning do begin
      SDLEvents.HandlePending;
      //Render;
      RenderOptimized;
    end;
  finally

    if FShowMarkers then begin
      Markers.Free;
      Markers := nil;
    end;

    //case LRenderMode of
    //  rendNormal: { do nothing };
    //  rendTestMode: SDL.App.Renderer.TestMode.Finalize;
    //end;


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

    //SDL_GL_DeleteContext(FSDLGLContext);
    SDL_DestroyRenderer(FSDLRenderer);
    SDL_DestroyWindow(FSDLWindow);
    SDL_Quit;
  end;

  //Print('Good Bye');
  if Assigned(OnClose) then
    OnClose(Self);
end;

procedure TSDLApplication.PrintRendererSetup;
var
  LSDLRendererInfo : TSDL_RendererInfo;

  function FlagsToNames(flags: Cardinal): string;
  begin
    Result := '';

    if (flags and SDL_RENDERER_SOFTWARE) <> 0 then
      Result := Result + 'SDL_RENDERER_SOFTWARE | ';
    if (flags and SDL_RENDERER_ACCELERATED) <> 0 then
      Result := Result + 'SDL_RENDERER_ACCELERATED | ';
    if (flags and SDL_RENDERER_PRESENTVSYNC) <> 0 then
      Result := Result + 'SDL_RENDERER_PRESENTVSYNC | ';
    if (flags and SDL_RENDERER_TARGETTEXTURE) <> 0 then
      Result := Result + 'SDL_RENDERER_TARGETTEXTURE | ';

    // Remove the trailing ' | ' if any
    if Result <> '' then
      Delete(Result, Length(Result) - 2, 3);
  end;
begin
  SDL_GetRendererInfo(FSDLRenderer, @LSDLRendererInfo);
  //Print(StrPas(LSDLRendererInfo.name)+' renderer initialized with flags:');
  //Print(FlagsToNames(LSDLRendererInfo.flags)+LineEnding)
end;

class procedure TSDLApplication.GetAvailableMonitors(AStrings: TStrings);
var
  i: Integer;
  LMonitors : array of TSDL_Rect = nil;
  LError: PAnsiChar;
begin
  if SDL_InitSubSystem(SDL_INIT_VIDEO) < 0 then begin
    LError := SDL_GetError;
    //Print(LError);
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
  //SDL_SetHint(SDL_HINT_RENDER_DRIVER, 'opengl');

  // setup monitor
  FCurrentMonitorIndex := AMonitor;
  LMonitor := FMonitors[AMonitor];

  // global opengl variables
  //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
  //SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  //SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 0);
  //SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

  // create sdl windows
  FSDLWindow := SDL_CreateWindow(FTitle, LMonitor.x, LMonitor.y,
    LMonitor.w, LMonitor.h, 0 {or SDL_WINDOW_OPENGL});

  // todo: create sdl opengl context
  // https://stackoverflow.com/questions/41091875/is-sdl-renderer-useless-if-i-use-opengl-for-drawing
  // FSDLGLContext := SDL_GL_CreateContext(FSDLWindow);

  // import opengl functions:
  // https://www.freepascal-meets-sdl.net/chapter-10-sdl-modern-opengl/
  // https://www.pascalgamedevelopment.com/archive/index.php/t-32663.html
  // https://wiki.libsdl.org/SDL2/SDL_GL_GetProcAddress
  // gladLoadGL(@SDL_GL_GetProcAddress);

  //if not Load_GL_VERSION_3_0_CORE then begin
  //  raise Exception.Create('Load_GL_VERSION_3_0_CORE error');
  //end;

  // create mouse handler
  Mouse := TSDLMouseHandler.Create(FSDLWindow);

  // create renderer
  FSDLRenderer := SDL_CreateRenderer(FSDLWindow, -1,
    SDL_RENDERER_ACCELERATED {or SDL_RENDERER_PRESENTVSYNC}
    //SDL_RENDERER_SOFTWARE
    );

  // expose window surface
  FSDLSurface  := SDL_GetWindowSurface(FSDLWindow);

  // expose global variables
  AssignVariables(FSDLWindow, FSDLRenderer, FSDLSurface);

  {$IFDEF NO_LCL}
  sdl.app.renderer.nolcl.Monitor := Self.Monitor;
  {$ELSE}

  {$ENDIF}
end;

procedure TSDLApplication.Terminate;
begin
  FRunning := False;
  //Print('Good bye');
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

