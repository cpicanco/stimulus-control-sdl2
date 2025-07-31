{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.controller.mouse;

{$mode ObjFPC}{$H+}

interface

uses sdl.app.controller;

type

  { TSDLMouseController }

  TSDLMouseController = class(TController)
    public
      procedure Hide; override;
      procedure Show; override;
  end;

implementation

uses session.parameters.global, sdl.app.mouse;

{ TSDLMouseController }

procedure TSDLMouseController.Hide;
begin
  Mouse.Hide;
end;

procedure TSDLMouseController.Show;
begin
  if GlobalTrialParameters.HideMouse then begin
    { do not show }
  end else begin
    Mouse.Show;
  end;
end;

end.

