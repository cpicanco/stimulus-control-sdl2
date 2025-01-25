{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.selectable.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses SDL2;

type

  { ISelectable }

  ISelectable = interface
    ['{3914BD43-1105-4C45-BD28-9F1709AC16AB}']
    function GetCustomName: string;
    function Origen : TSDL_Point;
    function GetBoundsRect : TSDL_Rect;
    procedure Select;
    procedure Confirm;
    procedure SetCustomName(AValue: string);
    property CustomName : string read GetCustomName write SetCustomName;
  end;

implementation

end.
