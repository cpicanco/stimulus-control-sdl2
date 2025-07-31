{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.renderer.nolcl;

{$mode ObjFPC}{$H+}


interface

uses SDL2;

procedure RenderOptimized;

implementation

uses
  Classes,
  SysUtils
  //, sdl.app.video.writer.windows
  , sdl.app.renderer.validation
  , sdl.app.renderer.types
  , sdl.app.renderer.variables
  , sdl.app.video.methods
  , sdl.app.task
  ;



procedure RenderOptimized;
begin
  if GPaintingInvalidated then begin
    GPaintingInvalidated := False;

    TKeyboardTask.Render;

    SDL_RenderPresent(PSDLRenderer);
  end;
  CheckSynchronize;
  SDL_Delay(DELTA_TIME);
end;

end.



