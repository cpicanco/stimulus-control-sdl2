unit sdl.app.mouse;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SDL2;

type

  { TSDLMouseHandler }

  TSDLMouseHandler = class
  private
    FPSDLWindow: PSDL_Window;
  public
    constructor Create(APSDLWindow: PSDL_Window);
    procedure MoveTo(APoint : TSDL_Point);
    function State(out APoint : TSDL_Point): Uint32;
  end;

var
  Mouse : TSDLMouseHandler;

implementation

{ TSDLMouseHandler }

constructor TSDLMouseHandler.Create(APSDLWindow: PSDL_Window);
begin
  FPSDLWindow := APSDLWindow;
end;

procedure TSDLMouseHandler.MoveTo(APoint: TSDL_Point);
begin
  SDL_WarpMouseInWindow(FPSDLWindow, APoint.X, APoint.Y);
end;

function TSDLMouseHandler.State(out APoint: TSDL_Point): Uint32;
begin
  Result := SDL_GetMouseState(@APoint.X, @APoint.Y);
end;

end.

