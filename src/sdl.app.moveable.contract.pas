{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.moveable.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses SDL2, sdl.app.events.abstract;

type

  IMoveable = interface
  ['{A57E409F-EC5E-4A9D-83A7-B69A32D8AE39}']
    function GetSDLMouseMotion: TOnMouseMotionEvent;
    function GetSDLMouseButtonDown: TOnMouseButtonDownEvent;
    function GetSDLMouseButtonUp: TOnMouseButtonUpEvent;
    function PointInside(SDLPoint : TSDL_Point) : Boolean;
    function GetBoundsRect : TSDL_Rect;
    function GetMouseInside : Boolean;
    procedure SetMouseInside(AValue : Boolean);
    procedure MouseDown(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer);
    procedure MouseMove(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer);
    procedure MouseUp(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer);
    procedure MouseEnter(Sender: TObject);
    procedure MouseExit(Sender: TObject);
    property MouseInside : Boolean read GetMouseInside write SetMouseInside;
  end;

implementation

end.

