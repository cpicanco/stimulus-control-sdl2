{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.clickable.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses SDL2, sdl.app.events.abstract;

type

  IClickable = interface
  ['{114591A9-5C68-46EA-ACCE-0A498A16FD9C}']
    function GetSDLMouseButtonDown: TOnMouseButtonDownEvent;
    function GetSDLMouseButtonUp: TOnMouseButtonUpEvent;
    function GetBoundsRect : TSDL_Rect;
    function PointInside(SDLPoint : TSDL_Point) : Boolean;
    procedure MouseDown(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer);
    procedure MouseUp(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer);
  end;

implementation

end.

