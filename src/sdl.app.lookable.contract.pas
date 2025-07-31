{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.lookable.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses
  SDL2, sdl.app.events.abstract;

type

  { ILookable }

  ILookable = interface
  ['{C6AB5C2C-2AA4-4FFD-9892-A368BD6C4E03}']
    function GetGazeInside: Boolean;
    function IsGazeInside(SDLPoint : TSDL_Point; AInflateFactor: UInt32 = 0) : Boolean;
    procedure GazeMove(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer);
    procedure GazeEnter(Sender: TObject);
    procedure GazeExit(Sender: TObject);
    procedure SetGazeInside(AValue: Boolean);
    property GazeInside : Boolean read GetGazeInside write SetGazeInside;
  end;

implementation

end.

