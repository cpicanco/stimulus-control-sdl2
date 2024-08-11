{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimuli.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses
  Classes,
  sdl.app.trials.types,
  sdl.app.navigable.contract,
  sdl.app.selectable.list;

type
  { IStimuli }

  IStimuli = interface
    ['{6B18F44A-7450-4871-A2BB-A109FC2ED005}']
    function AsIStimuli : IStimuli;
    function AsINavigable : INavigable;
    function CustomName : string;
    function GetTrial : TObject;
    function MyResult : TTrialResult;
    function ToData : string;
    function Header : string;
    function IsStarter : Boolean;
    function Selectables : TSelectables;
    procedure DoExpectedResponse;
    procedure Load(AParameters : TStringList; AParent : TObject);
    procedure Start;
    procedure Stop;
  end;

implementation

end.

