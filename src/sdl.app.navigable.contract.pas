{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.navigable.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses sdl.app.navigator.contract;

type

  INavigable = interface
  ['{8919BF2E-EE00-4873-ACF0-40222299A426}']
    procedure UpdateNavigator;
    procedure SetNavigator(ANavigator : ITableNavigator);
  end;

implementation

end.

