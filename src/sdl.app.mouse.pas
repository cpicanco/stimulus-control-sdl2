{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
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
    procedure Hide;
    procedure Show;
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

procedure TSDLMouseHandler.Hide;
begin
  SDL_ShowCursor(SDL_DISABLE);
end;

procedure TSDLMouseHandler.Show;
begin
  SDL_ShowCursor(SDL_ENABLE);
end;

end.

