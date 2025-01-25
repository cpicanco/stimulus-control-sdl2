{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimulus.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses Classes, SDL2, sdl.app.trials.contract, sdl.app.stimulus.types;

type

  { IStimulus }

  IStimulus = interface
    ['{9530ADCA-4CEB-48D4-9548-3FF55B13E5F7}']
    procedure Load(AParameters : TStringList;
      AParent : TObject; ARect: TSDL_Rect);
    procedure Start;
    procedure Stop;
    procedure DoResponse(AHuman : Boolean);
    function GetID: TStimulusID;
    function ToData : string;
  end;

implementation

end.

