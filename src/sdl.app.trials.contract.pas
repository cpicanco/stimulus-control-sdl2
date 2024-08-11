{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.trials.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses Classes
  , Session.Configuration
  , sdl.app.trials.types
  , sdl.app.stimuli.contract
  , sdl.app.paintable.contract;

type

  { ITrial }

  ITrial = interface
    ['{3652F678-82C6-4FDC-AC8B-E41CF4453138}']
    function ConsequenceDelay: Cardinal;
    function ConsequenceInterval: Cardinal;
    function GetOnTrialEnd: TNotifyEvent;
    function GetTrialConfiguration: TTrialConfiguration;
    function InterTrialInterval : Cardinal;
    function AsITrial : ITrial;
    function AsIPaintable : IPaintable;
    function GetIStimuli : IStimuli;
    function MyResult : TTrialResult;
    //function Header : string;
    function ToData : string;
    procedure DoExpectedResponse;
    procedure Reboot;
    procedure EndTrial;
    procedure Hide;
    procedure SetOnTrialEnd(ANotifyEvent: TNotifyEvent);
    procedure SetTrialConfiguration(TrialData: TTrialConfiguration);
    procedure Show;
    property Configuration : TTrialConfiguration read GetTrialConfiguration write SetTrialConfiguration;
    property OnTrialEnd : TNotifyEvent read GetOnTrialEnd write SetOnTrialEnd;
  end;

implementation

end.

