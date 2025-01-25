{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.choiceable.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

type

  { IChoiceable }

  IChoiceable = interface
    ['{B98EB54C-FA0B-47C5-87FF-82106015353E}']
    procedure AddOrderedChoice(AChoiceInstance : TObject);
    function GetTargetChoice: TObject;
    property TargetChoice : TObject read GetTargetChoice;
  end;
implementation

end.

