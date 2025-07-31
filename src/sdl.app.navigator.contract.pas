{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.navigator.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses
  sdl.app.selectable.contract,
  sdl.app.selectable.list,
  sdl.app.controller.types;

type

  { ILineNavigator }

  ILineNavigator = interface
    ['{DEBD72A2-BC97-4D1C-80AB-A0CAB06BAEFC}']
    procedure Unselect;
    procedure SelectNext;
    procedure SelectPrevious;
    procedure ConfirmSelection;
    procedure SetBaseControl(AControl : ISelectable);
    procedure UpdateNavigationControls(AControls : TSelectables);
  end;

  { ITableNavigator }

  ITableNavigator = interface
    ['{DEBD72A2-BC97-4D1C-80AB-A0CAB06BAEFC}']
    procedure Select;
    procedure SelectTarget(AControl : ISelectable);
    procedure GoTop;
    procedure GoBottom;
    procedure GoLeft;
    procedure GoRight;
    procedure GoTopRight;
    procedure GoBottomLeft;
    procedure GoTopLeft;
    procedure GoBottomRight;
    procedure GoBaseControl;
    //procedure GoLevelUp;
    //procedure GoLevelDown;
    procedure ConfirmSelection;
    procedure SetBaseControl(AControl : ISelectable);
    procedure UpdateNavigationControls(AControls : TSelectables);
  end;


implementation

end.
